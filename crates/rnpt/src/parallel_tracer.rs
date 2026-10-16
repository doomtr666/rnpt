use crate::{
    PathTracer, PathTracerConfig, Pixel, SamplingStrategy,
    evaluate_surface,
    nirc::{NircConfig, NircRingBuffer, NircTrainer},
};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

const TILE_SIZE: usize = 128;
const BATCH_SIZE: u64 = 256;
/// One out of every N rendered pixels triggers a dedicated MIS training path.
const TRAIN_PIXEL_STRIDE: usize = 64;
/// Workers refresh their cached NIRC network from the shared config this often (in pixels).
const NIRC_REFRESH_STRIDE: usize = 512;
/// Number of random samples read from the ring buffer per training call.
const RING_TRAIN_BATCH: usize = 512;
/// NIRC training is paced by rays, not time: one training step is due every
/// this many rays traced (closest-hit + shadow, training paths included).
const RAYS_PER_TRAIN_STEP: u64 = 1 << 18;
/// Workers wait when rendering gets this many steps ahead of training.
const MAX_TRAIN_LAG_STEPS: u64 = 2;
/// Upper bound on training steps run by one `train_nirc` call.
const MAX_TRAIN_STEPS_PER_CALL: u64 = 32;

/// A lock-free pixel buffer that multiple threads can write to concurrently.
/// This is safe because each worker thread writes to a unique disjoint set of pixels.
/// The GUI thread can also read from it concurrently, which might cause slight "tearing"
/// of an individual pixel's float values for one frame, but in a progressive path tracer
/// this creates absolutely zero visual artifacts.
struct UnsafePixelBuffer {
    ptr: *mut Pixel,
}

unsafe impl Send for UnsafePixelBuffer {}
unsafe impl Sync for UnsafePixelBuffer {}

impl UnsafePixelBuffer {
    fn new(ptr: *mut Pixel) -> Self {
        Self { ptr }
    }

    #[inline(always)]
    unsafe fn get_mut(&self, index: usize) -> &mut Pixel {
        unsafe { &mut *self.ptr.add(index) }
    }
}

/// See `ParallelTracer::render_nirc_probe`.
pub struct NircProbe {
    pub size: usize,
    /// Row-major, top row first.
    pub pixels: Vec<ProbePixel>,
    /// Where the surface normal points on the ball ([-1, 1], y up), if visible.
    pub normal: Option<[f32; 2]>,
}

#[derive(Clone, Copy)]
pub enum ProbePixel {
    /// Outside the ball.
    Outside,
    /// A direction below the surface.
    Below,
    /// Predicted incident radiance from that direction.
    Radiance([f32; 3]),
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TracerTiming {
    pub threads: usize,
    pub nirc_steps: u64,
    /// Time spent in `train_nirc` (on the caller's thread).
    pub train_s: f64,
    /// Time workers spent waiting for training, summed over workers.
    pub pacing_wait_s: f64,
}

pub struct ParallelTracer {
    config: Arc<Mutex<PathTracerConfig>>,
    /// Incremented on full scene changes (camera, geometry). Workers restart their pass.
    epoch: Arc<AtomicU32>,
    /// Incremented only when the NIRC network is updated. Workers hot-swap their network
    /// without restarting the render pass or recreating PathTracer.
    nirc_epoch: Arc<AtomicU32>,
    pixels: Vec<Pixel>,      // owned pixel buffer; threads hold raw pointers into this
    threads: Vec<thread::JoinHandle<()>>,

    // Performance metrics
    total_rays: Arc<AtomicU64>,        // paths (one per sample_pixel)
    total_real_rays: Arc<AtomicU64>,   // closest-hit rays (primary + bounces)
    total_shadow_rays: Arc<AtomicU64>, // any-hit shadow rays
    running: Arc<AtomicBool>,

    // NIRC training state
    /// Rays traced since the tracer was created; drives the training schedule.
    nirc_rays: Arc<AtomicU64>,
    /// Training steps run so far.
    nirc_steps: Arc<AtomicU64>,
    /// Time spent training, and summed over workers waiting for training (ns).
    train_ns: AtomicU64,
    pacing_wait_ns: Arc<AtomicU64>,
    num_threads: usize,
    /// Pixels that have spent their budget.
    pixels_done: Arc<AtomicU64>,
    nirc_trainer: Mutex<NircTrainer>,
    /// Lock-free ring buffer: workers push training samples, trainer reads random batches.
    nirc_ring: Arc<NircRingBuffer>,
}

impl ParallelTracer {
    pub fn new(config: PathTracerConfig) -> Self {
        let width = config.width;
        let height = config.height;
        let num_pixels = width * height;

        let mut pixels = vec![Pixel::default(); num_pixels];
        let buffer_ptr = pixels.as_mut_ptr();
        let shared_buffer = Arc::new(UnsafePixelBuffer::new(buffer_ptr));

        let nirc_ring = Arc::new(NircRingBuffer::new());

        let config = Arc::new(Mutex::new(config));
        let epoch = Arc::new(AtomicU32::new(1));
        let nirc_epoch = Arc::new(AtomicU32::new(0));
        let total_rays = Arc::new(AtomicU64::new(0));
        let total_real_rays = Arc::new(AtomicU64::new(0));
        let total_shadow_rays = Arc::new(AtomicU64::new(0));
        let running = Arc::new(AtomicBool::new(true));
        let nirc_rays = Arc::new(AtomicU64::new(0));
        let nirc_steps = Arc::new(AtomicU64::new(0));
        let pacing_wait_ns = Arc::new(AtomicU64::new(0));
        let pixels_done = Arc::new(AtomicU64::new(0));

        let num_threads = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8);
        let mut threads = Vec::with_capacity(num_threads);

        for thread_idx in 0..num_threads {
            let buffer = shared_buffer.clone();
            let ring = nirc_ring.clone();
            let config_mutex = config.clone();
            let epoch_atomic = epoch.clone();
            let nirc_epoch_atomic = nirc_epoch.clone();
            let rays_atomic = total_rays.clone();
            let real_rays_atomic = total_real_rays.clone();
            let shadow_rays_atomic = total_shadow_rays.clone();
            let running_atomic = running.clone();
            let nirc_rays_atomic = nirc_rays.clone();
            let nirc_steps_atomic = nirc_steps.clone();
            let pacing_wait_atomic = pacing_wait_ns.clone();
            let pixels_done_atomic = pixels_done.clone();

            threads.push(
                thread::Builder::new()
                    .name(format!("Worker-{}", thread_idx))
                    .stack_size(16 * 1024 * 1024)
                    .spawn(move || {
                        Self::worker_loop(
                            thread_idx,
                            num_threads,
                            buffer,
                            ring,
                            config_mutex,
                            epoch_atomic,
                            nirc_epoch_atomic,
                            rays_atomic,
                            real_rays_atomic,
                            shadow_rays_atomic,
                            running_atomic,
                            nirc_rays_atomic,
                            nirc_steps_atomic,
                            pacing_wait_atomic,
                            pixels_done_atomic,
                        )
                    })
                    .expect("Failed to spawn worker thread"),
            );
        }

        Self {
            config,
            epoch,
            nirc_epoch,
            pixels,
            threads,
            total_rays,
            total_real_rays,
            total_shadow_rays,
            running,
            nirc_rays,
            nirc_steps,
            train_ns: AtomicU64::new(0),
            pacing_wait_ns,
            num_threads,
            pixels_done,
            nirc_trainer: Mutex::new(NircTrainer::new(NircConfig::default())),
            nirc_ring,
        }
    }

    /// Train the NIRC from the ring buffer and publish the updated network to workers.
    ///
    /// Training is paced by rays, not time: one step is due every
    /// `RAYS_PER_TRAIN_STEP` rays traced, and workers wait when they get ahead
    /// (see `worker_loop`). Two renders with the same ray budget therefore train
    /// the network for the same number of steps, whatever the machine or the
    /// frame rate. Runs the steps due (at most `MAX_TRAIN_STEPS_PER_CALL`) and
    /// returns the loss of the last one, or `None` if none ran.
    pub fn train_nirc(&self) -> Option<f32> {
        if self.config.lock().unwrap().strategy != SamplingStrategy::Nirc {
            return None;
        }

        let mut trainer = match self.nirc_trainer.try_lock() {
            Ok(guard) => guard,
            Err(_) => return None,
        };

        let t0 = std::time::Instant::now();
        let due = self.nirc_rays.load(Ordering::Acquire) / RAYS_PER_TRAIN_STEP;
        let mut last_loss = None;
        for _ in 0..MAX_TRAIN_STEPS_PER_CALL {
            if self.nirc_steps.load(Ordering::Relaxed) >= due {
                break;
            }
            let samples = self.nirc_ring.read_random_batch(RING_TRAIN_BATCH);
            if !samples.is_empty() {
                last_loss = Some(trainer.train_samples(&samples));
            }
            // Counted even without samples, so workers never wait on an empty ring.
            self.nirc_steps.fetch_add(1, Ordering::Release);
        }
        self.train_ns.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let loss = last_loss?;

        // Publish the EMA snapshot. `ema_buf` is maintained in-place by `train_samples`;
        // we only allocate here (one Box per step) to hand off an immutable Arc to workers.
        let new_arc = Arc::from(trainer.ema_buf.clone_boxed());
        {
            let mut cfg = self.config.lock().unwrap();
            cfg.nirc_network = Some(new_arc);
        }
        self.nirc_epoch.fetch_add(1, Ordering::Release);
        Some(loss)
    }

    pub fn update_scene(&mut self, new_config: PathTracerConfig) {
        {
            let mut cfg = self.config.lock().unwrap();
            *cfg = new_config;
        }

        // If resolution changed, we can't resize `self.pixels` while threads are running
        // with raw pointers to it! We must recreate the ParallelTracer.
        // For simplicity, we assume `update_scene` is only called if resolution stays the same.
        // If it changes, the GUI should drop and recreate ParallelTracer.

        self.epoch.fetch_add(1, Ordering::SeqCst);
    }

    pub fn fetch_pixels(&self, target_buffer: &mut [Pixel]) {
        assert_eq!(target_buffer.len(), self.pixels.len());
        target_buffer.copy_from_slice(&self.pixels);
    }

    /// Number of training samples currently in the ring buffer.
    pub fn ring_filled(&self) -> usize {
        self.nirc_ring.filled()
    }


    /// Pixels that have spent their budget: cheap to poll, unlike fetching the image.
    pub fn pixels_done(&self) -> u64 {
        self.pixels_done.load(Ordering::Relaxed)
    }

    /// Where the time goes, for diagnosing NIRC's cost.
    pub fn timing(&self) -> TracerTiming {
        TracerTiming {
            threads: self.num_threads,
            nirc_steps: self.nirc_steps.load(Ordering::Relaxed),
            train_s: self.train_ns.load(Ordering::Relaxed) as f64 * 1e-9,
            pacing_wait_s: self.pacing_wait_ns.load(Ordering::Relaxed) as f64 * 1e-9,
        }
    }

    pub fn pop_rays_traced(&self) -> u64 {
        self.total_rays.swap(0, Ordering::Relaxed)
    }

    pub fn pop_real_rays_traced(&self) -> u64 {
        self.total_real_rays.swap(0, Ordering::Relaxed)
    }

    pub fn pop_shadow_rays_traced(&self) -> u64 {
        self.total_shadow_rays.swap(0, Ordering::Relaxed)
    }

    /// What the NIRC network predicts at the surface point seen through pixel
    /// (px, py), drawn as a small ball resting on the surface and seen from the
    /// camera: each point of the ball shows the radiance arriving from the
    /// direction it faces. `back` turns the view around to show the other half.
    /// `None` if no network yet or no surface hit.
    pub fn render_nirc_probe(&self, px: f32, py: f32, size: usize, back: bool) -> Option<NircProbe> {
        let cfg = self.config.lock().ok()?.clone();
        let network = cfg.nirc_network.as_ref()?.clone();
        let (bounds_min, bounds_max) = (cfg.bvh.bounds_min, cfg.bvh.bounds_max);

        // Same seed for every ray, hence the same sub-pixel jitter.
        let pt = PathTracer::new(cfg.clone());
        let ray_at = |x: f32, y: f32| pt.generate_ray(&mut crate::Pcg32::from_seed_128(42), x, y);
        let ray = ray_at(px, py);
        let hit = cfg.bvh.intersect(&ray)?;
        let surf = evaluate_surface(&hit, &ray, &cfg.bvh, &cfg.scene);
        let n = surf.normal.into_inner();

        // Screen axes in world space, taken from neighbouring camera rays so the
        // ball is oriented exactly like the image.
        let d = ray.direction.into_inner();
        let mut view = -d; // toward the camera
        let mut right = ray_at(px + 1.0, py).direction.into_inner() - d;
        right = (right - view * right.dot(&view)).normalize();
        let mut up = ray_at(px, py - 1.0).direction.into_inner() - d;
        up = (up - view * up.dot(&view) - right * up.dot(&right)).normalize();
        if back {
            // Walk around the ball: half a turn about the screen's up axis.
            view = -view;
            right = -right;
        }

        let mut pixels = Vec::with_capacity(size * size);
        for j in 0..size {
            for i in 0..size {
                let u = (i as f32 + 0.5) / size as f32 * 2.0 - 1.0;
                let v = 1.0 - (j as f32 + 0.5) / size as f32 * 2.0;
                let r2 = u * u + v * v;
                if r2 > 1.0 {
                    pixels.push(ProbePixel::Outside);
                    continue;
                }
                let wi = right * u + up * v + view * (1.0 - r2).sqrt();
                if wi.dot(&n) <= 0.0 {
                    // Below the surface: never queried at runtime.
                    pixels.push(ProbePixel::Below);
                    continue;
                }
                let input = crate::nirc::encode_inputs(&surf.position, &wi, &bounds_min, &bounds_max);
                let pred = network.forward(input);
                pixels.push(ProbePixel::Radiance([pred[0].max(0.0), pred[1].max(0.0), pred[2].max(0.0)]));
            }
        }

        // Where the surface normal points, on the ball (if on the visible half).
        let normal = (n.dot(&view) > 0.0).then(|| [n.dot(&right), n.dot(&up)]);

        Some(NircProbe { size, pixels, normal })
    }

    fn worker_loop(
        thread_idx: usize,
        num_threads: usize,
        buffer: Arc<UnsafePixelBuffer>,
        nirc_ring: Arc<NircRingBuffer>,
        config_mutex: Arc<Mutex<PathTracerConfig>>,
        epoch_atomic: Arc<AtomicU32>,
        nirc_epoch_atomic: Arc<AtomicU32>,
        rays_atomic: Arc<AtomicU64>,
        real_rays_atomic: Arc<AtomicU64>,
        shadow_rays_atomic: Arc<AtomicU64>,
        running_atomic: Arc<AtomicBool>,
        nirc_rays_atomic: Arc<AtomicU64>,
        nirc_steps_atomic: Arc<AtomicU64>,
        pacing_wait_atomic: Arc<AtomicU64>,
        pixels_done_atomic: Arc<AtomicU64>,
    ) {
        let mut local_epoch = 0u32;
        let mut local_nirc_epoch = u32::MAX; // force first refresh
        let mut path_tracer: Option<PathTracer> = None;
        let mut width = 0;
        let mut height = 0;
        let mut budget = crate::RenderBudget::default();
        let mut is_nirc = false;
        let mut train_rng =
            crate::Pcg32::from_seed_128(thread_idx as u128 * 0x9e3779b97f4a7c15);

        while running_atomic.load(Ordering::Relaxed) {
            let current_epoch = epoch_atomic.load(Ordering::Relaxed);

            if current_epoch != local_epoch {
                local_epoch = current_epoch;
                let cfg = config_mutex.lock().unwrap().clone();
                width = cfg.width;
                height = cfg.height;
                budget = cfg.budget;
                is_nirc = cfg.strategy == SamplingStrategy::Nirc;
                local_nirc_epoch = nirc_epoch_atomic.load(Ordering::Relaxed);
                path_tracer = Some(PathTracer::new(cfg));
            }

            let tracer = path_tracer.as_mut().unwrap();

            let mut rays_traced = 0u64;
            let mut real_rays_traced = 0u64;
            let mut shadow_rays_traced = 0u64;
            let mut pixel_count = 0usize;
            let mut did_work = false;

            let blocks_x = (width + TILE_SIZE - 1) / TILE_SIZE;
            let blocks_y = (height + TILE_SIZE - 1) / TILE_SIZE;
            let total_blocks = blocks_x * blocks_y;
            let base_count = total_blocks / num_threads;
            let remainder = total_blocks % num_threads;

            let (start_block, end_block) = if thread_idx < remainder {
                let start = thread_idx * (base_count + 1);
                (start, start + base_count + 1)
            } else {
                let start = remainder * (base_count + 1) + (thread_idx - remainder) * base_count;
                (start, start + base_count)
            };

            'pass: for block_idx in start_block..end_block {
                let bx = block_idx % blocks_x;
                let by = block_idx / blocks_x;

                let start_x = bx * TILE_SIZE;
                let start_y = by * TILE_SIZE;
                let end_x = (start_x + TILE_SIZE).min(width);
                let end_y = (start_y + TILE_SIZE).min(height);

                for y in start_y..end_y {
                    for x in start_x..end_x {
                        let i = y * width + x;

                        // Pixels that spent their budget are done; the others go on.
                        if budget.is_exhausted(unsafe { buffer.get_mut(i) }) {
                            continue;
                        }
                        did_work = true;

                        if rays_traced >= BATCH_SIZE {
                            if epoch_atomic.load(Ordering::Relaxed) != local_epoch
                                || !running_atomic.load(Ordering::Relaxed)
                            {
                                break 'pass;
                            }
                            rays_atomic.fetch_add(rays_traced, Ordering::Relaxed);
                            real_rays_atomic.fetch_add(real_rays_traced, Ordering::Relaxed);
                            shadow_rays_atomic.fetch_add(shadow_rays_traced, Ordering::Relaxed);
                            nirc_rays_atomic
                                .fetch_add(real_rays_traced + shadow_rays_traced, Ordering::Release);
                            rays_traced = 0;
                            real_rays_traced = 0;
                            shadow_rays_traced = 0;

                            // Pace rendering on training: never get more than
                            // MAX_TRAIN_LAG_STEPS steps ahead of the trainer.
                            while is_nirc
                                && nirc_rays_atomic.load(Ordering::Acquire)
                                    > (nirc_steps_atomic.load(Ordering::Acquire) + MAX_TRAIN_LAG_STEPS)
                                        * RAYS_PER_TRAIN_STEP
                            {
                                if epoch_atomic.load(Ordering::Relaxed) != local_epoch
                                    || !running_atomic.load(Ordering::Relaxed)
                                {
                                    break 'pass;
                                }
                                let t0 = std::time::Instant::now();
                                thread::sleep(std::time::Duration::from_micros(100));
                                pacing_wait_atomic
                                    .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                            }
                        }

                        // Hot-swap NIRC network without restarting the render pass.
                        pixel_count += 1;
                        if pixel_count % NIRC_REFRESH_STRIDE == 0 {
                            let cur = nirc_epoch_atomic.load(Ordering::Acquire);
                            if cur != local_nirc_epoch {
                                local_nirc_epoch = cur;
                                if let Ok(cfg) = config_mutex.try_lock() {
                                    tracer.set_nirc_network(cfg.nirc_network.clone());
                                }
                            }
                        }

                        let pixel = unsafe { buffer.get_mut(i) };
                        let (r, s) = tracer.sample_pixel(x, y, pixel);
                        real_rays_traced += r;
                        shadow_rays_traced += s;
                        rays_traced += 1;

                        // Training: random pixel subset (pixel_count stride gives diversity).
                        // Its rays are charged to the pixel: they are part of NIRC's cost.
                        if is_nirc && pixel_count % TRAIN_PIXEL_STRIDE == 0 {
                            let (mut tr, mut ts) = (0u64, 0u64);
                            let new_samples = tracer
                                .collect_training_samples_for_pixel(x, y, &mut train_rng, &mut tr, &mut ts);
                            for (inp, tgt) in new_samples {
                                nirc_ring.push(inp, tgt);
                            }
                            pixel.rays += tr + ts;
                            real_rays_traced += tr;
                            shadow_rays_traced += ts;
                        }
                        // Exhausted pixels are skipped above, so this counts each pixel once.
                        if budget.is_exhausted(pixel) {
                            pixels_done_atomic.fetch_add(1, Ordering::Relaxed);
                        }
                    }

                    if epoch_atomic.load(Ordering::Relaxed) != local_epoch
                        || !running_atomic.load(Ordering::Relaxed)
                    {
                        break 'pass;
                    }
                }
            }

            if rays_traced > 0 {
                rays_atomic.fetch_add(rays_traced, Ordering::Relaxed);
                real_rays_atomic.fetch_add(real_rays_traced, Ordering::Relaxed);
                shadow_rays_atomic.fetch_add(shadow_rays_traced, Ordering::Relaxed);
                nirc_rays_atomic.fetch_add(real_rays_traced + shadow_rays_traced, Ordering::Release);
            }

            // Every pixel of this worker spent its budget: idle until a reset.
            if !did_work {
                thread::sleep(std::time::Duration::from_millis(20));
            }
        }
    }
}

impl Drop for ParallelTracer {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}
