use nalgebra::SVector;
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrd};
use super::encoding::{INPUT_DIM, OUTPUT_DIM};

/// Lock-free circular buffer for NIRC training samples.
///
/// Multiple worker threads push concurrently without locks; the trainer thread
/// reads random batches. Occasional torn reads under contention are acceptable
/// for stochastic gradient training.
pub const RING_CAPACITY: usize = 1 << 17; // 131 072 entries ≈ 26 MB

pub struct NircRingBuffer {
    data: Vec<UnsafeCell<(SVector<f32, INPUT_DIM>, SVector<f32, OUTPUT_DIM>)>>,
    write_idx: AtomicUsize, // monotonically increasing
}

unsafe impl Send for NircRingBuffer {}
unsafe impl Sync for NircRingBuffer {}

impl NircRingBuffer {
    pub fn new() -> Self {
        Self {
            data: (0..RING_CAPACITY)
                .map(|_| UnsafeCell::new((SVector::zeros(), SVector::zeros())))
                .collect(),
            write_idx: AtomicUsize::new(0),
        }
    }

    #[inline]
    pub fn push(&self, input: SVector<f32, INPUT_DIM>, target: SVector<f32, OUTPUT_DIM>) {
        let idx = self.write_idx.fetch_add(1, AtomicOrd::Relaxed);
        let slot = idx & (RING_CAPACITY - 1);
        unsafe { *self.data[slot].get() = (input, target); }
    }

    /// Number of slots that have been written (capped at RING_CAPACITY).
    pub fn filled(&self) -> usize {
        self.write_idx.load(AtomicOrd::Relaxed).min(RING_CAPACITY)
    }

    /// Read `count` random samples from the filled portion.
    /// Seeded from the current write position — varies each call.
    pub fn read_random_batch(
        &self,
        count: usize,
    ) -> Vec<(SVector<f32, INPUT_DIM>, SVector<f32, OUTPUT_DIM>)> {
        let filled = self.filled();
        if filled == 0 {
            return Vec::new();
        }
        let seed = self.write_idx.load(AtomicOrd::Relaxed);
        let mut rng = crate::Pcg32::from_seed(seed as u64, 1);
        let n = count.min(filled);
        (0..n)
            .map(|_| {
                let slot = ((rng.next_f32() * filled as f32) as usize).min(filled - 1);
                unsafe { *self.data[slot].get() }
            })
            .collect()
    }
}
