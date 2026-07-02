use nalgebra::{SMatrix, SVector};
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrd};

pub struct NircConfig {
    pub learning_rate: f32,
    pub batch_size: usize,
    pub ema_alpha: f32,
}

impl Default for NircConfig {
    fn default() -> Self {
        Self {
            learning_rate: 3e-4,
            batch_size: 64,
            ema_alpha: 0.1,
        }
    }
}

pub struct NircTrainer {
    pub network: Box<NircMlp>,
    /// EMA-smoothed inference snapshot, updated in-place after each training step.
    pub ema_buf: Box<NircMlp>,
    /// Gradient accumulator — allocated once, zeroed at the start of each batch.
    grads: Box<NircMlpGradients>,
    pub optimizer: Box<AdamOptimizer>,
    pub config: NircConfig,
}

impl NircTrainer {
    pub fn new(config: NircConfig) -> Self {
        let network = NircMlp::new_boxed();
        let ema_buf = network.clone_boxed(); // EMA starts = initial weights (no cold-start bias)
        Self {
            ema_buf,
            network,
            grads: NircMlpGradients::new_boxed(),
            optimizer: AdamOptimizer::new_boxed(config.learning_rate),
            config,
        }
    }

    /// Trains on pre-collected samples. Returns the average loss over all batches.
    /// Updates `ema_buf` in-place after training (no allocation).
    pub fn train_samples(
        &mut self,
        samples: &[(nalgebra::SVector<f32, INPUT_DIM>, nalgebra::SVector<f32, 3>)],
    ) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let mut total_loss = 0.0f32;
        let mut num_batches = 0usize;
        for chunk in samples.chunks(self.config.batch_size) {
            total_loss += self
                .network
                .train_batch(&mut *self.optimizer, &mut *self.grads, chunk);
            num_batches += 1;
        }
        self.ema_buf.ema(&self.network, self.config.ema_alpha);
        if num_batches > 0 {
            total_loss / num_batches as f32
        } else {
            0.0
        }
    }
}

pub const POS_BINS: usize = 1;
/// Number of sinusoidal NeRF frequencies applied to each direction component.
/// Frequency k has period 2/2^k in [-1,1]; at k=9 the period is ~0.004 in dz
/// → angular resolution ~3.6° even near the poles of the unit sphere.
pub const DIR_FREQS: usize = 1;
pub const INPUT_DIM: usize = POS_BINS * 3 + DIR_FREQS * 2 * 3;
const HIDDEN_DIM: usize = 16;
const OUTPUT_DIM: usize = 3;

pub struct NircForwardCache {
    x: SVector<f32, INPUT_DIM>,
    z1: SVector<f32, HIDDEN_DIM>,
    a1: SVector<f32, HIDDEN_DIM>,
    z2: SVector<f32, HIDDEN_DIM>,
    a2: SVector<f32, HIDDEN_DIM>,
    z3: SVector<f32, HIDDEN_DIM>,
    a3: SVector<f32, HIDDEN_DIM>,
}

#[derive(Debug)]
pub struct NircMlpGradients {
    w1: SMatrix<f32, HIDDEN_DIM, INPUT_DIM>,
    b1: SVector<f32, HIDDEN_DIM>,
    w2: SMatrix<f32, HIDDEN_DIM, HIDDEN_DIM>,
    b2: SVector<f32, HIDDEN_DIM>,
    w3: SMatrix<f32, HIDDEN_DIM, HIDDEN_DIM>,
    b3: SVector<f32, HIDDEN_DIM>,
    w4: SMatrix<f32, OUTPUT_DIM, HIDDEN_DIM>,
    b4: SVector<f32, OUTPUT_DIM>,
}

impl NircMlpGradients {
    pub fn zeros() -> Self {
        NircMlpGradients {
            w1: SMatrix::zeros(),
            b1: SVector::zeros(),
            w2: SMatrix::zeros(),
            b2: SVector::zeros(),
            w3: SMatrix::zeros(),
            b3: SVector::zeros(),
            w4: SMatrix::zeros(),
            b4: SVector::zeros(),
        }
    }

    /// Allocates zeroed gradients on the heap.
    pub fn new_boxed() -> Box<Self> {
        unsafe {
            let layout = std::alloc::Layout::new::<Self>();
            let ptr = std::alloc::alloc_zeroed(layout) as *mut Self;
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            Box::from_raw(ptr)
        }
    }

    /// Zeros all gradient values in-place. Used to reset a pre-allocated buffer.
    pub fn zero_in_place(&mut self) {
        unsafe {
            std::ptr::write_bytes(self as *mut Self as *mut u8, 0, std::mem::size_of::<Self>());
        }
    }

    pub fn add_gradients(&mut self, other: &Self) {
        self.w1 += other.w1;
        self.b1 += other.b1;
        self.w2 += other.w2;
        self.b2 += other.b2;
        self.w3 += other.w3;
        self.b3 += other.b3;
        self.w4 += other.w4;
        self.b4 += other.b4;
    }

    pub fn divide_by(&mut self, n: f32) {
        self.w1 /= n;
        self.b1 /= n;
        self.w2 /= n;
        self.b2 /= n;
        self.w3 /= n;
        self.b3 /= n;
        self.w4 /= n;
        self.b4 /= n;
    }
}

/// Ad hoc multi level perceptron used in the radiance cache
#[derive(Debug)]
pub struct NircMlp {
    w1: SMatrix<f32, HIDDEN_DIM, INPUT_DIM>,
    b1: SVector<f32, HIDDEN_DIM>,

    w2: SMatrix<f32, HIDDEN_DIM, HIDDEN_DIM>,
    b2: SVector<f32, HIDDEN_DIM>,

    w3: SMatrix<f32, HIDDEN_DIM, HIDDEN_DIM>,
    b3: SVector<f32, HIDDEN_DIM>,

    w4: SMatrix<f32, OUTPUT_DIM, HIDDEN_DIM>,
    b4: SVector<f32, OUTPUT_DIM>,
}

impl NircMlp {
    /// Allocate on the heap and initialize weights (Xavier uniform). No stack intermediate —
    /// SMatrix<f32,256,256> is 256 KB; constructing it via `Self { w: SMatrix::from_fn(...) }`
    /// would put that on the caller's stack. Instead we alloc zeroed then write in-place.
    pub fn new_boxed() -> Box<Self> {
        unsafe {
            let layout = std::alloc::Layout::new::<Self>();
            let ptr = std::alloc::alloc_zeroed(layout) as *mut Self;
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            let b = &mut *ptr;

            let mut rng = crate::Pcg32::from_seed_128(42);
            let limit_hidden = (6.0f32 / (HIDDEN_DIM + HIDDEN_DIM) as f32).sqrt();
            let limit_in = (6.0f32 / (HIDDEN_DIM + INPUT_DIM) as f32).sqrt();
            let limit_out = (6.0f32 / (OUTPUT_DIM + HIDDEN_DIM) as f32).sqrt();

            for v in b.w1.as_mut_slice() {
                *v = (rng.next_f32() * 2.0 - 1.0) * limit_in;
            }
            for v in b.w2.as_mut_slice() {
                *v = (rng.next_f32() * 2.0 - 1.0) * limit_hidden;
            }
            for v in b.w3.as_mut_slice() {
                *v = (rng.next_f32() * 2.0 - 1.0) * limit_hidden;
            }
            for v in b.w4.as_mut_slice() {
                *v = (rng.next_f32() * 2.0 - 1.0) * limit_out;
            }
            // biases remain zero

            Box::from_raw(ptr)
        }
    }

    /// Copy onto the heap without any stack intermediate.
    /// NircMlp is ~625 KB — memcpy via pointer, never materialised on the stack.
    pub fn clone_boxed(&self) -> Box<Self> {
        unsafe {
            let layout = std::alloc::Layout::new::<Self>();
            let ptr = std::alloc::alloc(layout) as *mut Self;
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            std::ptr::copy_nonoverlapping(self as *const Self, ptr, 1);
            Box::from_raw(ptr)
        }
    }

    /// Exponential Moving Average update
    pub fn ema(&mut self, other: &Self, alpha: f32) {
        let beta = 1.0 - alpha;
        // Element-wise indexing: zero stack temporaries.
        // The naive `self.w2 = self.w2 * beta + other.w2 * alpha` creates up to three
        // 256×256 matrix temporaries (768 KB) on the stack, overflowing the main thread.
        macro_rules! ema_field {
            ($dst:expr, $src:expr) => {
                for i in 0..$dst.len() {
                    $dst[i] = $dst[i] * beta + $src[i] * alpha;
                }
            };
        }
        ema_field!(self.w1, other.w1);
        ema_field!(self.b1, other.b1);
        ema_field!(self.w2, other.w2);
        ema_field!(self.b2, other.b2);
        ema_field!(self.w3, other.w3);
        ema_field!(self.b3, other.b3);
        ema_field!(self.w4, other.w4);
        ema_field!(self.b4, other.b4);
    }

    /// Applique le Positional Encoding (Encodage Positionnel) classique NeRF + coordonnées brutes.
    /// Transforme une position 3D et une direction 3D en un vecteur de dimensions `INPUT_DIM`.
    pub fn encode_inputs(
        pos: &nalgebra::Point3<f32>,
        dir: &nalgebra::Vector3<f32>,
        bounds_min: &nalgebra::Point3<f32>,
        bounds_max: &nalgebra::Point3<f32>,
    ) -> SVector<f32, INPUT_DIM> {
        let mut input = SVector::<f32, INPUT_DIM>::zeros();
        let mut idx = 0;

        let extents = bounds_max - bounds_min;
        // Normalize position to [0, 1]
        let norm_pos = nalgebra::Point3::new(
            if extents.x > 0.0 {
                (pos.x - bounds_min.x) / extents.x
            } else {
                0.5
            },
            if extents.y > 0.0 {
                (pos.y - bounds_min.y) / extents.y
            } else {
                0.5
            },
            if extents.z > 0.0 {
                (pos.z - bounds_min.z) / extents.z
            } else {
                0.5
            },
        );
        // Oneblob position encoding
        let pos_sigma = 1.0 / (POS_BINS as f32);
        let pos_inv_2sigma2 = (POS_BINS as f32).powi(2) / 2.0;

        for i in 0..POS_BINS {
            let c = (i as f32 + 0.5) * pos_sigma;
            input[idx] = (-(norm_pos.x - c).powi(2) * pos_inv_2sigma2).exp();
            input[idx + 1] = (-(norm_pos.y - c).powi(2) * pos_inv_2sigma2).exp();
            input[idx + 2] = (-(norm_pos.z - c).powi(2) * pos_inv_2sigma2).exp();
            idx += 3;
        }

        // NeRF sinusoidal direction encoding.
        // dir components are in [-1, 1] (unit vector). Each frequency k encodes
        // sin(2^k π d) and cos(2^k π d) for each component, giving 2*3 features per k.
        // High frequencies (k≥8) resolve sub-degree angular features near any pole.
        for k in 0..DIR_FREQS {
            let freq = (1u32 << k) as f32 * std::f32::consts::PI;
            input[idx] = (freq * dir.x).sin();
            input[idx + 1] = (freq * dir.y).sin();
            input[idx + 2] = (freq * dir.z).sin();
            idx += 3;
            input[idx] = (freq * dir.x).cos();
            input[idx + 1] = (freq * dir.y).cos();
            input[idx + 2] = (freq * dir.z).cos();
            idx += 3;
        }

        input
    }

    /// Runs one gradient step on `batch`. `grads` is a pre-allocated buffer (owned by
    /// `NircTrainer`) — it is zeroed here and reused across calls, so no allocation occurs.
    pub fn train_batch(
        &mut self,
        opt: &mut AdamOptimizer,
        grads: &mut NircMlpGradients,
        batch: &[(SVector<f32, INPUT_DIM>, SVector<f32, 3>)],
    ) -> f32 {
        if batch.is_empty() {
            return 0.0;
        }
        grads.zero_in_place();
        let mut batch_loss = 0.0f32;
        for (input, target) in batch {
            let (pred, cache) = self.forward_for_training(*input);
            let dl_dy = Self::compute_loss_derivative(&pred, target);
            batch_loss += Self::compute_loss(&pred, target);
            self.backward_into(&cache, dl_dy, grads);
        }
        grads.divide_by(batch.len() as f32);
        opt.step(self, grads);
        batch_loss / batch.len() as f32
    }

    /// Log-space MSE: L = (log(1+pred⁺) - log(1+target))²
    /// Balanced gradient weight across the full HDR dynamic range:
    /// gradient ∝ 1/(1+pred), so bright samples (ceiling light) are not ignored.
    #[inline]
    pub fn compute_loss(pred: &SVector<f32, OUTPUT_DIM>, target: &SVector<f32, OUTPUT_DIM>) -> f32 {
        let mut loss = 0.0f32;
        for i in 0..OUTPUT_DIM {
            let target_log = (1.0 + target[i]).ln();
            let pred_log = (1.0 + pred[i].max(0.0)).ln();
            let err = pred_log - target_log;
            loss += err * err;
        }
        loss / OUTPUT_DIM as f32
    }

    /// Activation function: SiLU (Sigmoid Linear Unit)
    /// f(x) = x * sigmoid(x)
    #[inline(always)]
    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    /// Derivative of SiLU
    /// f'(x) = f(x) + sigmoid(x) * (1 - f(x))
    #[inline(always)]
    fn silu_deriv(x: f32) -> f32 {
        let sig = 1.0 / (1.0 + (-x).exp());
        let f = x * sig;
        f + sig * (1.0 - f)
    }

    /// Applies SiLU derivative element-wise to a vector
    #[inline(always)]
    fn apply_silu_deriv<const D: usize>(v: &SVector<f32, D>) -> SVector<f32, D> {
        v.map(Self::silu_deriv)
    }

    /// Pure inference forward pass.
    /// Uses `gemv` (BLAS matrix-vector product, takes A by reference) so the weight
    /// matrices are never copied. The `*` operator on SMatrix: Copy would silently
    /// copy 111–256 KB per layer (626 KB total), causing the VCRUNTIME memcpy hotspot.
    /// Two 256-f32 accumulators alternate; each is initialised to the layer bias so
    /// gemv(alpha, W, x, beta=1) folds the bias add into the single kernel call.
    pub fn forward(&self, x: SVector<f32, INPUT_DIM>) -> SVector<f32, OUTPUT_DIM> {
        let mut a = self.b1.clone_owned();
        a.gemv(1.0, &self.w1, &x, 1.0);
        for v in a.iter_mut() {
            *v = Self::silu(*v);
        }

        let mut b = self.b2.clone_owned();
        b.gemv(1.0, &self.w2, &a, 1.0);
        for v in b.iter_mut() {
            *v = Self::silu(*v);
        }

        a.copy_from(&self.b3);
        a.gemv(1.0, &self.w3, &b, 1.0);
        for v in a.iter_mut() {
            *v = Self::silu(*v);
        }

        let mut out = self.b4.clone_owned();
        out.gemv(1.0, &self.w4, &a, 1.0);
        out
    }

    /// Training forward pass.
    /// Returns the final RGB output and the cache needed for backpropagation.
    /// Use this ONLY for the 5% of training rays.
    pub fn forward_for_training(
        &self,
        x: SVector<f32, INPUT_DIM>,
    ) -> (SVector<f32, OUTPUT_DIM>, NircForwardCache) {
        let mut z1 = self.b1.clone_owned();
        z1.gemv(1.0, &self.w1, &x, 1.0);
        let a1 = z1.map(Self::silu);

        let mut z2 = self.b2.clone_owned();
        z2.gemv(1.0, &self.w2, &a1, 1.0);
        let a2 = z2.map(Self::silu);

        let mut z3 = self.b3.clone_owned();
        z3.gemv(1.0, &self.w3, &a2, 1.0);
        let a3 = z3.map(Self::silu);

        let mut z4 = self.b4.clone_owned();
        z4.gemv(1.0, &self.w4, &a3, 1.0);

        (
            z4,
            NircForwardCache {
                x,
                z1,
                a1,
                z2,
                a2,
                z3,
                a3,
            },
        )
    }

    /// Backward pass: accumulates gradients into `grads` (does not allocate).
    /// Weight gradients use `ger()` (rank-1 update, no temporary matrix).
    /// Error signals use `gemv_tr` (computes A^T * x without materialising A^T,
    /// avoiding 256 KB copies of w2/w3 that `wN.transpose() * delta` would incur).
    pub fn backward_into(
        &self,
        cache: &NircForwardCache,
        dl_dy: SVector<f32, OUTPUT_DIM>,
        grads: &mut NircMlpGradients,
    ) {
        // SVector<f32, N>: Copy — each `+= delta` copies the vector, so `delta`
        // remains valid for the subsequent gemv_tr call.
        let delta4 = dl_dy;
        grads.w4.ger(1.0, &delta4, &cache.a3, 1.0);
        grads.b4 += delta4;

        let sp3 = Self::apply_silu_deriv(&cache.z3);
        let mut delta3 = SVector::<f32, HIDDEN_DIM>::zeros();
        delta3.gemv_tr(1.0, &self.w4, &delta4, 0.0);
        delta3.component_mul_assign(&sp3);
        grads.w3.ger(1.0, &delta3, &cache.a2, 1.0);
        grads.b3 += delta3;

        let sp2 = Self::apply_silu_deriv(&cache.z2);
        let mut delta2 = SVector::<f32, HIDDEN_DIM>::zeros();
        delta2.gemv_tr(1.0, &self.w3, &delta3, 0.0);
        delta2.component_mul_assign(&sp2);
        grads.w2.ger(1.0, &delta2, &cache.a1, 1.0);
        grads.b2 += delta2;

        let sp1 = Self::apply_silu_deriv(&cache.z1);
        let mut delta1 = SVector::<f32, HIDDEN_DIM>::zeros();
        delta1.gemv_tr(1.0, &self.w2, &delta2, 0.0);
        delta1.component_mul_assign(&sp1);
        grads.w1.ger(1.0, &delta1, &cache.x, 1.0);
        grads.b1 += delta1;
    }

    /// Gradient of log-space MSE w.r.t. the network output (linear radiance).
    /// dL/dpred = (log(1+pred⁺) - log(1+target)) / (1+pred⁺)
    /// For pred ≤ 0: linear approximation d/dpred(log(1+pred⁺))≈1 avoids gradient kill.
    pub fn compute_loss_derivative(
        pred: &SVector<f32, OUTPUT_DIM>,
        target: &SVector<f32, OUTPUT_DIM>,
    ) -> SVector<f32, OUTPUT_DIM> {
        let mut dl_dy = SVector::<f32, OUTPUT_DIM>::zeros();
        for i in 0..OUTPUT_DIM {
            let target_log = (1.0 + target[i]).ln();
            if pred[i] > 0.0 {
                let pred_log = (1.0 + pred[i]).ln();
                dl_dy[i] = (pred_log - target_log) / (1.0 + pred[i]);
            } else {
                // pred ≤ 0: treat log(1+pred⁺) = 0, d(·)/dpred = 1
                dl_dy[i] = -target_log;
            }
        }
        dl_dy
    }
}

/// Adam Optimizer state.
/// Holds the first moment (m) and second moment (v) for every parameter in the network.
#[derive(Debug)]
pub struct AdamOptimizer {
    pub m: NircMlpGradients,
    pub v: NircMlpGradients,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub lr: f32,
    pub t: u32,
}

impl AdamOptimizer {
    /// Allocates the optimizer on the heap. `AdamOptimizer` holds two `NircMlpGradients`
    /// inline (m and v, ~620 KB each → ~1.24 MB total). `alloc_zeroed` avoids constructing
    /// them on the caller's stack; the scalar fields are written through the pointer.
    pub fn new_boxed(learning_rate: f32) -> Box<Self> {
        unsafe {
            let layout = std::alloc::Layout::new::<Self>();
            let ptr = std::alloc::alloc_zeroed(layout) as *mut Self;
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            let b = &mut *ptr;
            // m and v are zeroed by alloc_zeroed — that is their correct initial state.
            b.beta1 = 0.9;
            b.beta2 = 0.999;
            b.epsilon = 1e-8;
            b.lr = learning_rate;
            b.t = 0;
            Box::from_raw(ptr)
        }
    }

    /// Applies the accumulated gradients to the network's weights and biases.
    pub fn step(&mut self, network: &mut NircMlp, grads: &NircMlpGradients) {
        self.t += 1;

        // Bias correction factors
        let b1_t = 1.0 - self.beta1.powi(self.t as i32);
        let b2_t = 1.0 - self.beta2.powi(self.t as i32);

        // Precompute the effective learning rate to save cycles
        let lr_t = self.lr * (b2_t.sqrt() / b1_t);

        // Helper macro to apply the Adam update rule element-wise
        // LLVM easily unrolls and vectorizes this flat indexing loop.
        macro_rules! update_param {
            ($param:expr, $m:expr, $v:expr, $grad:expr) => {
                for i in 0..$param.len() {
                    $m[i] = self.beta1 * $m[i] + (1.0 - self.beta1) * $grad[i];
                    $v[i] = self.beta2 * $v[i] + (1.0 - self.beta2) * $grad[i] * $grad[i];

                    $param[i] -= lr_t * $m[i] / ($v[i].sqrt() + self.epsilon);
                }
            };
        }

        update_param!(network.w1, self.m.w1, self.v.w1, grads.w1);
        update_param!(network.b1, self.m.b1, self.v.b1, grads.b1);
        update_param!(network.w2, self.m.w2, self.v.w2, grads.w2);
        update_param!(network.b2, self.m.b2, self.v.b2, grads.b2);
        update_param!(network.w3, self.m.w3, self.v.w3, grads.w3);
        update_param!(network.b3, self.m.b3, self.v.b3, grads.b3);
        update_param!(network.w4, self.m.w4, self.v.w4, grads.w4);
        update_param!(network.b4, self.m.b4, self.v.b4, grads.b4);
    }
}

/// Lock-free circular buffer for NIRC training samples.
///
/// Multiple worker threads push concurrently without locks; the trainer thread
/// reads random batches from it. Occasional torn reads under contention are
/// acceptable for stochastic gradient training.
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
        unsafe {
            *self.data[slot].get() = (input, target);
        }
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
