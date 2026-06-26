use nalgebra::{SMatrix, SVector};

const INPUT_DIM: usize = 64;
const HIDDEN_DIM: usize = 64;
const OUTPUT_DIM: usize = 64;

pub struct NrcForwardCache {
    x: SVector<f32, INPUT_DIM>,
    z1: SVector<f32, HIDDEN_DIM>,
    a1: SVector<f32, HIDDEN_DIM>,
    z2: SVector<f32, HIDDEN_DIM>,
    a2: SVector<f32, HIDDEN_DIM>,
    z3: SVector<f32, HIDDEN_DIM>,
    a3: SVector<f32, HIDDEN_DIM>,
    z4: SVector<f32, HIDDEN_DIM>,
}

#[derive(Debug)]
pub struct NrcMlpGradients {
    w1: SMatrix<f32, HIDDEN_DIM, INPUT_DIM>,
    b1: SVector<f32, HIDDEN_DIM>,
    w2: SMatrix<f32, HIDDEN_DIM, HIDDEN_DIM>,
    b2: SVector<f32, HIDDEN_DIM>,
    w3: SMatrix<f32, HIDDEN_DIM, HIDDEN_DIM>,
    b3: SVector<f32, HIDDEN_DIM>,
    w4: SMatrix<f32, OUTPUT_DIM, HIDDEN_DIM>,
    b4: SVector<f32, OUTPUT_DIM>,
}

impl NrcMlpGradients {
    pub fn zeros() -> Self {
        NrcMlpGradients {
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
}

/// Ad hoc multi level perceptron used in the radiance cache
pub struct NrcMlp {
    w1: SMatrix<f32, HIDDEN_DIM, INPUT_DIM>,
    b1: SVector<f32, HIDDEN_DIM>,

    w2: SMatrix<f32, HIDDEN_DIM, HIDDEN_DIM>,
    b2: SVector<f32, HIDDEN_DIM>,

    w3: SMatrix<f32, HIDDEN_DIM, HIDDEN_DIM>,
    b3: SVector<f32, HIDDEN_DIM>,

    w4: SMatrix<f32, OUTPUT_DIM, HIDDEN_DIM>,
    b4: SVector<f32, OUTPUT_DIM>,
}

impl NrcMlp {
    /// Activation function: SiLU (Sigmoid Linear Unit)
    /// f(x) = x * sigmoid(x)
    #[inline(always)]
    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    /// Applies SiLU element-wise to a vector
    fn apply_silu<const D: usize>(v: &SVector<f32, D>) -> SVector<f32, D> {
        v.map(Self::silu)
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
    /// Use this for the 95% of rays that do not contribute to training.
    pub fn forward(&self, x: SVector<f32, INPUT_DIM>) -> SVector<f32, OUTPUT_DIM> {
        let a1 = Self::apply_silu(&(self.w1 * x + self.b1));
        let a2 = Self::apply_silu(&(self.w2 * a1 + self.b2));
        let a3 = Self::apply_silu(&(self.w3 * a2 + self.b3));

        // Output layer (linear)
        self.w4 * a3 + self.b4
    }

    /// Training forward pass.
    /// Returns the final RGB output and the cache needed for backpropagation.
    /// Use this ONLY for the 5% of training rays.
    pub fn forward_for_training(
        &self,
        x: SVector<f32, INPUT_DIM>,
    ) -> (SVector<f32, OUTPUT_DIM>, NrcForwardCache) {
        // Layer 1
        let z1 = self.w1 * x + self.b1;
        let a1 = Self::apply_silu(&z1);

        // Layer 2
        let z2 = self.w2 * a1 + self.b2;
        let a2 = Self::apply_silu(&z2);

        // Layer 3
        let z3 = self.w3 * a2 + self.b3;
        let a3 = Self::apply_silu(&z3);

        // Layer 4 (Output) - Linear activation for raw radiance values.
        let z4 = self.w4 * a3 + self.b4;

        let cache = NrcForwardCache {
            x,
            z1,
            a1,
            z2,
            a2,
            z3,
            a3,
            z4,
        };

        (z4, cache)
    }

    /// Backward pass computing the gradients.
    /// dl_dy is the gradient of the loss with respect to the network's output.
    pub fn backward(
        &self,
        cache: &NrcForwardCache,
        dl_dy: SVector<f32, OUTPUT_DIM>,
    ) -> NrcMlpGradients {
        // Layer 4 (Output)
        // Since there is no activation function on the output, dL_dz4 == dL_dy
        let delta4 = dl_dy;
        let grad_w4 = delta4 * cache.a3.transpose();
        let grad_b4 = delta4;

        // Layer 3
        let sp3 = Self::apply_silu_deriv(&cache.z3);
        let delta3 = (self.w4.transpose() * delta4).component_mul(&sp3);
        let grad_w3 = delta3 * cache.a2.transpose();
        let grad_b3 = delta3;

        // Layer 2
        let sp2 = Self::apply_silu_deriv(&cache.z2);
        let delta2 = (self.w3.transpose() * delta3).component_mul(&sp2);
        let grad_w2 = delta2 * cache.a1.transpose();
        let grad_b2 = delta2;

        // Layer 1
        let sp1 = Self::apply_silu_deriv(&cache.z1);
        let delta1 = (self.w2.transpose() * delta2).component_mul(&sp1);
        let grad_w1 = delta1 * cache.x.transpose();
        let grad_b1 = delta1;

        NrcMlpGradients {
            w1: grad_w1,
            b1: grad_b1,
            w2: grad_w2,
            b2: grad_b2,
            w3: grad_w3,
            b3: grad_b3,
            w4: grad_w4,
            b4: grad_b4,
        }
    }

    /// Computes the loss derivative dL/dy using a Relative L2 Loss with stop-gradient.
    /// target: the raw HDR radiance from the path tracer (y).
    /// pred: the radiance predicted by the active network (\hat{y}).
    pub fn compute_loss_derivative(
        pred: &SVector<f32, OUTPUT_DIM>,
        target: &SVector<f32, OUTPUT_DIM>,
    ) -> SVector<f32, OUTPUT_DIM> {
        let mut dl_dy = SVector::<f32, OUTPUT_DIM>::zeros();

        // Epsilon prevents division by zero in completely dark areas.
        // 1e-2 is the standard empirical value for radiance caching.
        let epsilon = 1e-2;

        for i in 0..OUTPUT_DIM {
            let diff = pred[i] - target[i];

            // Stop-gradient applied here: we compute the normalizer using the current
            // prediction, but we do not derive this term. It acts as a static weight
            // for the current backward pass.
            let normalizer = pred[i] * pred[i] + epsilon;

            // The mathematical factor of 2.0 is intentionally dropped as it simply
            // scales the global learning rate.
            dl_dy[i] = diff / normalizer;
        }

        dl_dy
    }
}

/// Adam Optimizer state.
/// Holds the first moment (m) and second moment (v) for every parameter in the network.
#[derive(Debug)]
pub struct AdamOptimizer {
    pub m: NrcMlpGradients,
    pub v: NrcMlpGradients,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub lr: f32,
    pub t: u32,
}

impl AdamOptimizer {
    /// Creates a new Adam optimizer with default parameters.

    pub fn new(learning_rate: f32) -> Self {
        Self {
            m: NrcMlpGradients::zeros(),
            v: NrcMlpGradients::zeros(),
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            lr: learning_rate,
            t: 0,
        }
    }

    /// Applies the accumulated gradients to the network's weights and biases.
    pub fn step(&mut self, network: &mut NrcMlp, grads: &NrcMlpGradients) {
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

pub struct NeuralRadianceCache {}
