use crate::{
    PathTracer, PathTracerConfig, Pixel, SamplingStrategy,
    evaluate_surface,
    nirc::{NircConfig, NircMlp, NircRingBuffer, NircTrainer},
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
            nirc_trainer: Mutex::new(NircTrainer::new(NircConfig::default())),
            nirc_ring,
        }
    }

    /// Train the NIRC from the ring buffer and publish the updated network to workers.
    /// Returns the average training loss for this step, or `None` if skipped.
    pub fn train_nirc(&self) -> Option<f32> {
        if self.config.lock().unwrap().strategy != SamplingStrategy::Nirc {
            return None;
        }

        let samples = self.nirc_ring.read_random_batch(RING_TRAIN_BATCH);
        if samples.is_empty() {
            return None;
        }

        let mut trainer = match self.nirc_trainer.try_lock() {
            Ok(guard) => guard,
            Err(_) => return None,
        };

        let loss = trainer.train_samples(&samples);

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


    pub fn pop_rays_traced(&self) -> u64 {
        self.total_rays.swap(0, Ordering::Relaxed)
    }

    pub fn pop_real_rays_traced(&self) -> u64 {
        self.total_real_rays.swap(0, Ordering::Relaxed)
    }

    pub fn pop_shadow_rays_traced(&self) -> u64 {
        self.total_shadow_rays.swap(0, Ordering::Relaxed)
    }

    /// Render an equirectangular map of what the NIRC network predicts from the
    /// surface point hit by pixel (px, py). Returns `None` if no network is loaded
    /// or the primary ray doesn't hit any geometry.
    pub fn render_nirc_probe(
        &self,
        px: f32,
        py: f32,
        probe_w: usize,
        probe_h: usize,
    ) -> Option<Vec<[f32; 3]>> {
        let cfg = self.config.lock().ok()?.clone();
        let network = cfg.nirc_network.as_ref()?.clone();
        let bounds_min = cfg.bvh.bounds_min;
        let bounds_max = cfg.bvh.bounds_max;

        let pt = PathTracer::new(cfg.clone());
        let mut rng = crate::Pcg32::from_seed_128(42);
        let ray = pt.generate_ray(&mut rng, px, py);

        let hit = cfg.bvh.intersect(&ray)?;
        let surf = evaluate_surface(&hit, &ray, &cfg.bvh, &cfg.scene);
        let pos = surf.position;
        let geo_normal = surf.geo_normal;

        let mut probe = Vec::with_capacity(probe_w * probe_h);
        for row in 0..probe_h {
            for col in 0..probe_w {
                let phi = (col as f32 + 0.5) / probe_w as f32 * std::f32::consts::TAU;
                let theta = (row as f32 + 0.5) / probe_h as f32 * std::f32::consts::PI;
                let wi = nalgebra::Vector3::new(
                    theta.sin() * phi.cos(),
                    theta.cos(),
                    theta.sin() * phi.sin(),
                );

                // Below the geometric horizon the network is never queried at runtime.
                if wi.dot(geo_normal.as_ref()) <= 0.0 {
                    probe.push([0.0, 0.0, 0.0]);
                    continue;
                }

                let input = NircMlp::encode_inputs(&pos, &wi, &bounds_min, &bounds_max);
                let pred = network.forward(input);
                probe.push([pred[0].max(0.0), pred[1].max(0.0), pred[2].max(0.0)]);
            }
        }
        Some(probe)
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
    ) {
        let mut local_epoch = 0u32;
        let mut local_nirc_epoch = u32::MAX; // force first refresh
        let mut path_tracer: Option<PathTracer> = None;
        let mut width = 0;
        let mut height = 0;
        let mut train_rng =
            crate::Pcg32::from_seed_128(thread_idx as u128 * 0x9e3779b97f4a7c15);

        while running_atomic.load(Ordering::Relaxed) {
            let current_epoch = epoch_atomic.load(Ordering::Relaxed);

            if current_epoch != local_epoch {
                local_epoch = current_epoch;
                let cfg = config_mutex.lock().unwrap().clone();
                width = cfg.width;
                height = cfg.height;
                local_nirc_epoch = nirc_epoch_atomic.load(Ordering::Relaxed);
                path_tracer = Some(PathTracer::new(cfg));
            }

            let tracer = path_tracer.as_mut().unwrap();

            let mut rays_traced = 0u64;
            let mut real_rays_traced = 0u64;
            let mut shadow_rays_traced = 0u64;
            let mut pixel_count = 0usize;

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

                        if rays_traced >= BATCH_SIZE {
                            if epoch_atomic.load(Ordering::Relaxed) != local_epoch
                                || !running_atomic.load(Ordering::Relaxed)
                            {
                                break 'pass;
                            }
                            rays_atomic.fetch_add(rays_traced, Ordering::Relaxed);
                            real_rays_atomic.fetch_add(real_rays_traced, Ordering::Relaxed);
                            shadow_rays_atomic.fetch_add(shadow_rays_traced, Ordering::Relaxed);
                            rays_traced = 0;
                            real_rays_traced = 0;
                            shadow_rays_traced = 0;
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

                        unsafe {
                            let pixel = buffer.get_mut(i);
                            let (r, s) = tracer.sample_pixel(x, y, pixel);
                            real_rays_traced += r;
                            shadow_rays_traced += s;


                        }
                        rays_traced += 1;

                        // Every TRAIN_PIXEL_STRIDE pixels, trace a dedicated MIS path
                        // and push all bounce samples into the lock-free ring buffer.
                        if pixel_count % TRAIN_PIXEL_STRIDE == 0 {
                            let new_samples = tracer
                                .collect_training_samples_for_pixel(x, y, &mut train_rng);
                            for (inp, tgt) in new_samples {
                                nirc_ring.push(inp, tgt);
                            }
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
