use crate::{
    PathTracer, PathTracerConfig, Pixel,
    nirc::{NircConfig, NircTrainer, INPUT_DIM},
};
use nalgebra::SVector;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

const TILE_SIZE: usize = 128;
const BATCH_SIZE: u64 = 256;
/// One out of every N rendered pixels triggers a full training path (returns multiple samples).
/// Higher than before because each path yields one sample per bounce (Cornell ≈ 4-6 samples/path).
const TRAIN_PIXEL_STRIDE: usize = 64;
/// Worker-local sample capacity before flushing to the shared buffer (one short lock).
const LOCAL_TRAIN_BATCH_CAPACITY: usize = 64;
/// Workers refresh their cached NIRC network from the shared config this often (in pixels).
const NIRC_REFRESH_STRIDE: usize = 512;
/// Hard cap on the shared sample buffer. If workers fill it faster than train_nirc can drain,
/// we discard the excess to prevent unbounded growth and O(n) training cost per frame.
const MAX_TRAINING_BUFFER: usize = 4096;

type NircSample = (SVector<f32, INPUT_DIM>, SVector<f32, 3>);

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
    pixels: Vec<Pixel>, // owned pixel buffer; threads hold raw pointers into this
    threads: Vec<thread::JoinHandle<()>>,

    // Performance metrics
    total_rays: Arc<AtomicU64>,        // paths (one per sample_pixel)
    total_real_rays: Arc<AtomicU64>,   // closest-hit rays (primary + bounces)
    total_shadow_rays: Arc<AtomicU64>, // any-hit shadow rays
    running: Arc<AtomicBool>,

    // NIRC training state
    nirc_trainer: Mutex<NircTrainer>,
    /// Training samples collected inline by workers; drained by train_nirc.
    shared_sample_buffer: Arc<Mutex<Vec<NircSample>>>,
}

impl ParallelTracer {
    pub fn new(config: PathTracerConfig) -> Self {
        let width = config.width;
        let height = config.height;
        let num_pixels = width * height;

        let mut pixels = vec![Pixel::default(); num_pixels];
        let buffer_ptr = pixels.as_mut_ptr();
        let shared_buffer = Arc::new(UnsafePixelBuffer::new(buffer_ptr));

        let config = Arc::new(Mutex::new(config));
        let epoch = Arc::new(AtomicU32::new(1));
        let nirc_epoch = Arc::new(AtomicU32::new(0));
        let total_rays = Arc::new(AtomicU64::new(0));
        let total_real_rays = Arc::new(AtomicU64::new(0));
        let total_shadow_rays = Arc::new(AtomicU64::new(0));
        let running = Arc::new(AtomicBool::new(true));
        let shared_sample_buffer: Arc<Mutex<Vec<NircSample>>> =
            Arc::new(Mutex::new(Vec::new()));

        let num_threads = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8);
        let mut threads = Vec::with_capacity(num_threads);

        for thread_idx in 0..num_threads {
            let buffer = shared_buffer.clone();
            let config_mutex = config.clone();
            let epoch_atomic = epoch.clone();
            let nirc_epoch_atomic = nirc_epoch.clone();
            let rays_atomic = total_rays.clone();
            let real_rays_atomic = total_real_rays.clone();
            let shadow_rays_atomic = total_shadow_rays.clone();
            let running_atomic = running.clone();
            let sample_buffer = shared_sample_buffer.clone();

            threads.push(
                thread::Builder::new()
                    .name(format!("Worker-{}", thread_idx))
                    .stack_size(16 * 1024 * 1024)
                    .spawn(move || {
                        Self::worker_loop(
                            thread_idx,
                            num_threads,
                            buffer,
                            config_mutex,
                            epoch_atomic,
                            nirc_epoch_atomic,
                            rays_atomic,
                            real_rays_atomic,
                            shadow_rays_atomic,
                            running_atomic,
                            sample_buffer,
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
            shared_sample_buffer,
        }
    }

    /// Consumes samples collected by workers, trains the NIRC, then pushes the updated
    /// network to the shared config via EMA. Signals `nirc_epoch` so workers hot-swap
    /// the network without restarting their render pass.
    pub fn train_nirc(&self) {
        if self.config.lock().unwrap().strategy != crate::SamplingStrategy::Nirc {
            return;
        }

        // Drain the shared sample buffer; cap to avoid unbounded growth.
        let samples = {
            let mut buf = match self.shared_sample_buffer.try_lock() {
                Ok(g) => g,
                Err(_) => return,
            };
            if buf.is_empty() {
                return;
            }
            let mut v = std::mem::take(&mut *buf);
            if v.len() > MAX_TRAINING_BUFFER {
                v.drain(..v.len() - MAX_TRAINING_BUFFER);
            }
            v
        };

        let mut trainer = match self.nirc_trainer.try_lock() {
            Ok(guard) => guard,
            Err(_) => return,
        };

        trainer.train_samples(&samples);

        // EMA update: blend the inference network toward the freshly trained weights.
        let new_active = if let Some(old) = &self.config.lock().unwrap().nirc_network {
            let mut updated = (**old).clone();
            updated.ema(&*trainer.network, 0.05);
            updated
        } else {
            (*trainer.network).clone()
        };

        {
            let mut cfg_lock = self.config.lock().unwrap();
            cfg_lock.nirc_network = Some(Arc::new(new_active));
        }
        // Signal workers to hot-swap the network. Does NOT restart the render pass.
        self.nirc_epoch.fetch_add(1, Ordering::Release);
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
        // Concurrent read from the vector while threads are writing.
        // This is practically safe here since f32 tearing on X/Y/Z doesn't crash
        // and visually looks fine.
        target_buffer.copy_from_slice(&self.pixels);
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

    fn worker_loop(
        thread_idx: usize,
        num_threads: usize,
        buffer: Arc<UnsafePixelBuffer>,
        config_mutex: Arc<Mutex<PathTracerConfig>>,
        epoch_atomic: Arc<AtomicU32>,
        nirc_epoch_atomic: Arc<AtomicU32>,
        rays_atomic: Arc<AtomicU64>,
        real_rays_atomic: Arc<AtomicU64>,
        shadow_rays_atomic: Arc<AtomicU64>,
        running_atomic: Arc<AtomicBool>,
        sample_buffer: Arc<Mutex<Vec<NircSample>>>,
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
            let mut local_train: Vec<NircSample> =
                Vec::with_capacity(LOCAL_TRAIN_BATCH_CAPACITY);
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

                        // Every TRAIN_PIXEL_STRIDE pixels, trace a full unbiased path and
                        // collect one training sample per opaque bounce (suffix backward pass).
                        if pixel_count % TRAIN_PIXEL_STRIDE == 0 {
                            let new_samples = tracer
                                .collect_training_samples_for_pixel(x, y, &mut train_rng);
                            local_train.extend(new_samples);

                            if local_train.len() >= LOCAL_TRAIN_BATCH_CAPACITY {
                                if let Ok(mut buf) = sample_buffer.try_lock() {
                                    buf.extend(local_train.drain(..));
                                } else {
                                    local_train.clear(); // lock busy → discard, don't block
                                }
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

            // Flush remaining samples at end of pass.
            if !local_train.is_empty() {
                if let Ok(mut buf) = sample_buffer.try_lock() {
                    buf.extend(local_train.drain(..));
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
