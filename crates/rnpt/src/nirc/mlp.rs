use nalgebra::{SMatrix, SVector};
use super::encoding::{HIDDEN_DIM, INPUT_DIM, OUTPUT_DIM};

// ── Forward-pass cache ────────────────────────────────────────────────────────

pub struct NircForwardCache {
    pub(super) x:  SVector<f32, INPUT_DIM>,
    pub(super) z1: SVector<f32, HIDDEN_DIM>,
    pub(super) a1: SVector<f32, HIDDEN_DIM>,
    pub(super) z2: SVector<f32, HIDDEN_DIM>,
    pub(super) a2: SVector<f32, HIDDEN_DIM>,
    pub(super) z3: SVector<f32, HIDDEN_DIM>,
    pub(super) a3: SVector<f32, HIDDEN_DIM>,
}

// ── Gradient accumulator ──────────────────────────────────────────────────────

#[derive(Debug)]
pub struct NircMlpGradients {
    pub(super) w1: SMatrix<f32, HIDDEN_DIM, INPUT_DIM>,
    pub(super) b1: SVector<f32, HIDDEN_DIM>,
    pub(super) w2: SMatrix<f32, HIDDEN_DIM, HIDDEN_DIM>,
    pub(super) b2: SVector<f32, HIDDEN_DIM>,
    pub(super) w3: SMatrix<f32, HIDDEN_DIM, HIDDEN_DIM>,
    pub(super) b3: SVector<f32, HIDDEN_DIM>,
    pub(super) w4: SMatrix<f32, OUTPUT_DIM, HIDDEN_DIM>,
    pub(super) b4: SVector<f32, OUTPUT_DIM>,
}

impl NircMlpGradients {
    pub fn zeros() -> Self {
        NircMlpGradients {
            w1: SMatrix::zeros(), b1: SVector::zeros(),
            w2: SMatrix::zeros(), b2: SVector::zeros(),
            w3: SMatrix::zeros(), b3: SVector::zeros(),
            w4: SMatrix::zeros(), b4: SVector::zeros(),
        }
    }

    /// Allocate zeroed on the heap — avoids large stack temporaries.
    pub fn new_boxed() -> Box<Self> {
        unsafe {
            let layout = std::alloc::Layout::new::<Self>();
            let ptr = std::alloc::alloc_zeroed(layout) as *mut Self;
            if ptr.is_null() { std::alloc::handle_alloc_error(layout); }
            Box::from_raw(ptr)
        }
    }

    pub fn zero_in_place(&mut self) {
        unsafe {
            std::ptr::write_bytes(self as *mut Self as *mut u8, 0, std::mem::size_of::<Self>());
        }
    }

    pub fn add_gradients(&mut self, other: &Self) {
        self.w1 += other.w1; self.b1 += other.b1;
        self.w2 += other.w2; self.b2 += other.b2;
        self.w3 += other.w3; self.b3 += other.b3;
        self.w4 += other.w4; self.b4 += other.b4;
    }

    pub fn divide_by(&mut self, n: f32) {
        self.w1 /= n; self.b1 /= n;
        self.w2 /= n; self.b2 /= n;
        self.w3 /= n; self.b3 /= n;
        self.w4 /= n; self.b4 /= n;
    }
}

// ── MLP ───────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct NircMlp {
    pub(super) w1: SMatrix<f32, HIDDEN_DIM, INPUT_DIM>,
    pub(super) b1: SVector<f32, HIDDEN_DIM>,
    pub(super) w2: SMatrix<f32, HIDDEN_DIM, HIDDEN_DIM>,
    pub(super) b2: SVector<f32, HIDDEN_DIM>,
    pub(super) w3: SMatrix<f32, HIDDEN_DIM, HIDDEN_DIM>,
    pub(super) b3: SVector<f32, HIDDEN_DIM>,
    pub(super) w4: SMatrix<f32, OUTPUT_DIM, HIDDEN_DIM>,
    pub(super) b4: SVector<f32, OUTPUT_DIM>,
}

impl NircMlp {
    /// Allocate on the heap and initialize weights (Xavier uniform).
    /// Avoids large stack temporaries — SMatrix<f32,256,256> would be 256 KB on the stack.
    pub fn new_boxed() -> Box<Self> {
        unsafe {
            let layout = std::alloc::Layout::new::<Self>();
            let ptr = std::alloc::alloc_zeroed(layout) as *mut Self;
            if ptr.is_null() { std::alloc::handle_alloc_error(layout); }
            let b = &mut *ptr;

            let mut rng = crate::Pcg32::from_seed_128(42);
            let limit_in     = (6.0f32 / (HIDDEN_DIM + INPUT_DIM) as f32).sqrt();
            let limit_hidden = (6.0f32 / (HIDDEN_DIM + HIDDEN_DIM) as f32).sqrt();
            let limit_out    = (6.0f32 / (OUTPUT_DIM + HIDDEN_DIM) as f32).sqrt();

            for v in b.w1.as_mut_slice() { *v = (rng.next_f32() * 2.0 - 1.0) * limit_in; }
            for v in b.w2.as_mut_slice() { *v = (rng.next_f32() * 2.0 - 1.0) * limit_hidden; }
            for v in b.w3.as_mut_slice() { *v = (rng.next_f32() * 2.0 - 1.0) * limit_hidden; }
            for v in b.w4.as_mut_slice() { *v = (rng.next_f32() * 2.0 - 1.0) * limit_out; }
            // biases remain zero

            Box::from_raw(ptr)
        }
    }

    /// Heap-copy without any stack intermediate (~625 KB via memcpy).
    pub fn clone_boxed(&self) -> Box<Self> {
        unsafe {
            let layout = std::alloc::Layout::new::<Self>();
            let ptr = std::alloc::alloc(layout) as *mut Self;
            if ptr.is_null() { std::alloc::handle_alloc_error(layout); }
            std::ptr::copy_nonoverlapping(self as *const Self, ptr, 1);
            Box::from_raw(ptr)
        }
    }

    /// Exponential Moving Average: self = self*(1-α) + other*α (element-wise, no temporaries).
    pub fn ema(&mut self, other: &Self, alpha: f32) {
        let beta = 1.0 - alpha;
        macro_rules! ema_field {
            ($dst:expr, $src:expr) => {
                for i in 0..$dst.len() { $dst[i] = $dst[i] * beta + $src[i] * alpha; }
            };
        }
        ema_field!(self.w1, other.w1); ema_field!(self.b1, other.b1);
        ema_field!(self.w2, other.w2); ema_field!(self.b2, other.b2);
        ema_field!(self.w3, other.w3); ema_field!(self.b3, other.b3);
        ema_field!(self.w4, other.w4); ema_field!(self.b4, other.b4);
    }

    // ── Activation ───────────────────────────────────────────────────────────

    #[inline(always)]
    fn silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }

    #[inline(always)]
    fn silu_deriv(x: f32) -> f32 {
        let sig = 1.0 / (1.0 + (-x).exp());
        let f = x * sig;
        f + sig * (1.0 - f)
    }

    #[inline(always)]
    fn apply_silu_deriv<const D: usize>(v: &SVector<f32, D>) -> SVector<f32, D> {
        v.map(Self::silu_deriv)
    }

    // ── Forward passes ───────────────────────────────────────────────────────

    /// Pure inference forward pass. Uses `gemv` to avoid copying weight matrices.
    pub fn forward(&self, x: SVector<f32, INPUT_DIM>) -> SVector<f32, OUTPUT_DIM> {
        let mut a = self.b1.clone_owned();
        a.gemv(1.0, &self.w1, &x, 1.0);
        for v in a.iter_mut() { *v = Self::silu(*v); }

        let mut b = self.b2.clone_owned();
        b.gemv(1.0, &self.w2, &a, 1.0);
        for v in b.iter_mut() { *v = Self::silu(*v); }

        a.copy_from(&self.b3);
        a.gemv(1.0, &self.w3, &b, 1.0);
        for v in a.iter_mut() { *v = Self::silu(*v); }

        let mut out = self.b4.clone_owned();
        out.gemv(1.0, &self.w4, &a, 1.0);
        out
    }

    /// Training forward pass — caches intermediate activations for backprop.
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

        (z4, NircForwardCache { x, z1, a1, z2, a2, z3, a3 })
    }

    // ── Backward pass ────────────────────────────────────────────────────────

    /// Accumulates gradients into `grads` without allocating.
    /// Uses `ger()` for weight gradients and `gemv_tr` for error signals.
    pub fn backward_into(
        &self,
        cache: &NircForwardCache,
        dl_dy: SVector<f32, OUTPUT_DIM>,
        grads: &mut NircMlpGradients,
    ) {
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

    // ── Loss ─────────────────────────────────────────────────────────────────

    /// Log-space MSE: L = (log(1+pred⁺) − log(1+target))²
    /// Balanced gradient weight across HDR dynamic range: ∝ 1/(1+pred).
    #[inline]
    pub fn compute_loss(
        pred: &SVector<f32, OUTPUT_DIM>,
        target: &SVector<f32, OUTPUT_DIM>,
    ) -> f32 {
        let mut loss = 0.0f32;
        for i in 0..OUTPUT_DIM {
            let err = (1.0 + pred[i].max(0.0)).ln() - (1.0 + target[i]).ln();
            loss += err * err;
        }
        loss / OUTPUT_DIM as f32
    }

    /// dL/dpred_i = (log(1+pred⁺) − log(1+target)) / (1+pred⁺).
    /// For pred ≤ 0: linear approximation prevents gradient kill.
    pub fn compute_loss_derivative(
        pred: &SVector<f32, OUTPUT_DIM>,
        target: &SVector<f32, OUTPUT_DIM>,
    ) -> SVector<f32, OUTPUT_DIM> {
        let mut dl_dy = SVector::<f32, OUTPUT_DIM>::zeros();
        for i in 0..OUTPUT_DIM {
            let target_log = (1.0 + target[i]).ln();
            dl_dy[i] = if pred[i] > 0.0 {
                let pred_log = (1.0 + pred[i]).ln();
                (pred_log - target_log) / (1.0 + pred[i])
            } else {
                -target_log // treat log(1+pred⁺) = 0, d(·)/dpred = 1
            };
        }
        dl_dy
    }
}
