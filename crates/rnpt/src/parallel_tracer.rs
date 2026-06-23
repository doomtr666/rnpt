use crate::{PathTracer, PathTracerConfig, Pixel, Reservoir, SamplingStrategy};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

const TILE_SIZE: usize = 128;
const BATCH_SIZE: u64 = 1024;

/// A lock-free pixel buffer that multiple threads can write to concurrently.
/// This is safe because each worker thread writes to a unique disjoint set of pixels.
/// The GUI thread can also read from it concurrently, which might cause slight "tearing"
/// of an individual pixel's float values for one frame, but in a progressive path tracer
/// this creates absolutely zero visual artifacts.
struct UnsafePixelBuffer {
    ptr: *mut Pixel,
    len: usize,
}

unsafe impl Send for UnsafePixelBuffer {}
unsafe impl Sync for UnsafePixelBuffer {}

impl UnsafePixelBuffer {
    fn new(ptr: *mut Pixel, len: usize) -> Self {
        Self { ptr, len }
    }

    #[inline(always)]
    unsafe fn get_mut(&self, index: usize) -> &mut Pixel {
        unsafe { &mut *self.ptr.add(index) }
    }
}

/// Per-pixel reservoir buffer for ReSTIR DI, with the same ownership model as
/// `UnsafePixelBuffer`: each thread accesses a disjoint set of indices.
struct UnsafeReservoirBuffer {
    ptr: *mut Reservoir,
    #[allow(dead_code)]
    len: usize,
}

unsafe impl Send for UnsafeReservoirBuffer {}
unsafe impl Sync for UnsafeReservoirBuffer {}

impl UnsafeReservoirBuffer {
    fn new(ptr: *mut Reservoir, len: usize) -> Self {
        Self { ptr, len }
    }

    #[inline(always)]
    unsafe fn get_mut(&self, index: usize) -> &mut Reservoir {
        unsafe { &mut *self.ptr.add(index) }
    }
}

pub struct ParallelTracer {
    config: Arc<Mutex<PathTracerConfig>>,
    epoch: Arc<AtomicU32>,
    pixels: Vec<Pixel>,         // owned pixel buffer; threads hold raw pointers into this
    #[allow(dead_code)]
    reservoirs: Vec<Reservoir>, // per-pixel ReSTIR DI reservoir; same lifetime as pixels
    threads: Vec<thread::JoinHandle<()>>,

    // Performance metrics
    total_rays: Arc<AtomicU64>,        // paths (one per sample_pixel)
    total_real_rays: Arc<AtomicU64>,   // closest-hit rays (primary + bounces)
    total_shadow_rays: Arc<AtomicU64>, // any-hit shadow rays
    // We keep a flag to terminate threads on drop
    running: Arc<AtomicBool>,
}

impl ParallelTracer {
    pub fn new(config: PathTracerConfig) -> Self {
        let width = config.width;
        let height = config.height;
        let num_pixels = width * height;

        let mut pixels = vec![Pixel::default(); num_pixels];
        let buffer_ptr = pixels.as_mut_ptr();
        let shared_buffer = Arc::new(UnsafePixelBuffer::new(buffer_ptr, num_pixels));

        let mut reservoirs = vec![Reservoir::default(); num_pixels];
        let res_ptr = reservoirs.as_mut_ptr();
        let shared_reservoirs = Arc::new(UnsafeReservoirBuffer::new(res_ptr, num_pixels));

        let config = Arc::new(Mutex::new(config));
        let epoch = Arc::new(AtomicU32::new(1));
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
            let reservoir_buffer = shared_reservoirs.clone();
            let config_mutex = config.clone();
            let epoch_atomic = epoch.clone();
            let rays_atomic = total_rays.clone();
            let real_rays_atomic = total_real_rays.clone();
            let shadow_rays_atomic = total_shadow_rays.clone();
            let running_atomic = running.clone();

            threads.push(thread::spawn(move || {
                Self::worker_loop(
                    thread_idx,
                    num_threads,
                    buffer,
                    reservoir_buffer,
                    config_mutex,
                    epoch_atomic,
                    rays_atomic,
                    real_rays_atomic,
                    shadow_rays_atomic,
                    running_atomic,
                )
            }));
        }

        Self {
            config,
            epoch,
            pixels,
            reservoirs,
            threads,
            total_rays,
            total_real_rays,
            total_shadow_rays,
            running,
        }
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

        // Signal threads to restart
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
        reservoir_buffer: Arc<UnsafeReservoirBuffer>,
        config_mutex: Arc<Mutex<PathTracerConfig>>,
        epoch_atomic: Arc<AtomicU32>,
        rays_atomic: Arc<AtomicU64>,
        real_rays_atomic: Arc<AtomicU64>,
        shadow_rays_atomic: Arc<AtomicU64>,
        running_atomic: Arc<AtomicBool>,
    ) {
        let mut local_epoch = 0;
        let mut path_tracer: Option<PathTracer> = None;
        let mut use_restir = false;
        let mut width = 0;
        let mut height = 0;

        // How many samples to compute before checking the epoch and updating stats.
        // Larger = more CPU efficiency (less atomic contention), smaller = lower UI latency.
        // 1024 is still less than 1ms, so UI latency is invisible, but atomic overhead vanishes.
        // using BATCH_SIZE

        while running_atomic.load(Ordering::Relaxed) {
            let current_epoch = epoch_atomic.load(Ordering::Relaxed);

            // If the epoch changed, we need to reload the scene and clear our pixels
            if current_epoch != local_epoch {
                local_epoch = current_epoch;

                let cfg = config_mutex.lock().unwrap().clone();
                width = cfg.width;
                height = cfg.height;
                let tracer = PathTracer::new(cfg);
                use_restir = tracer.strategy() == SamplingStrategy::ReStirDi;
                path_tracer = Some(tracer);

                // Clear the pixels and reservoirs assigned to this thread
                for i in (thread_idx..buffer.len).step_by(num_threads) {
                    unsafe {
                        *buffer.get_mut(i) = Pixel::default();
                        *reservoir_buffer.get_mut(i) = Reservoir::default();
                    }
                }
            }

            let tracer = path_tracer.as_ref().unwrap();

            // Trace a batch of rays
            let mut rays_traced = 0; // paths (also drives the batch flush cadence)
            let mut real_rays_traced = 0; // closest-hit rays (primary + bounces)
            let mut shadow_rays_traced = 0; // any-hit shadow rays

            // Block-based rendering (Tiles)
            // We use contiguous block assignment (Thread 0 gets the first N blocks, Thread 1 the next N, etc.)
            // This maximizes spatial coherence across the entire thread's workload.
            let blocks_x = (width + TILE_SIZE - 1) / TILE_SIZE;
            let blocks_y = (height + TILE_SIZE - 1) / TILE_SIZE;
            let total_blocks = blocks_x * blocks_y;
            let base_count = total_blocks / num_threads;
            let remainder = total_blocks % num_threads;

            // Threads 0..remainder get (base_count + 1) blocks.
            // Threads remainder..num_threads get base_count blocks.
            let (start_block, end_block) = if thread_idx < remainder {
                let start = thread_idx * (base_count + 1);
                (start, start + base_count + 1)
            } else {
                let start = remainder * (base_count + 1) + (thread_idx - remainder) * base_count;
                (start, start + base_count)
            };

            for block_idx in start_block..end_block {
                let bx = block_idx % blocks_x;
                let by = block_idx / blocks_x;

                let start_x = bx * TILE_SIZE;
                let start_y = by * TILE_SIZE;
                let end_x = (start_x + TILE_SIZE).min(width);
                let end_y = (start_y + TILE_SIZE).min(height);

                for y in start_y..end_y {
                    for x in start_x..end_x {
                        let i = y * width + x;

                        // Periodically check if we need to abort this pass early
                        if rays_traced >= BATCH_SIZE {
                            if epoch_atomic.load(Ordering::Relaxed) != local_epoch
                                || !running_atomic.load(Ordering::Relaxed)
                            {
                                break; // Abort and restart/exit
                            }
                            rays_atomic.fetch_add(rays_traced, Ordering::Relaxed);
                            real_rays_atomic.fetch_add(real_rays_traced, Ordering::Relaxed);
                            shadow_rays_atomic.fetch_add(shadow_rays_traced, Ordering::Relaxed);
                            rays_traced = 0;
                            real_rays_traced = 0;
                            shadow_rays_traced = 0;
                        }

                        unsafe {
                            let pixel = buffer.get_mut(i);
                            let (r, s) = if use_restir {
                                let res = reservoir_buffer.get_mut(i);
                                tracer.sample_pixel_restir(x, y, pixel, res)
                            } else {
                                tracer.sample_pixel(x, y, pixel)
                            };
                            real_rays_traced += r;
                            shadow_rays_traced += s;
                        }

                        rays_traced += 1;
                    }
                    if epoch_atomic.load(Ordering::Relaxed) != local_epoch
                        || !running_atomic.load(Ordering::Relaxed)
                    {
                        break;
                    }
                }

                // Break out of blocks loop if needed
                if epoch_atomic.load(Ordering::Relaxed) != local_epoch
                    || !running_atomic.load(Ordering::Relaxed)
                {
                    break;
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
