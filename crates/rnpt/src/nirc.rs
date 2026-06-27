use nalgebra::{SMatrix, SVector};

pub struct NircConfig {
    pub learning_rate: f32,
    pub batch_size: usize,
}

impl Default for NircConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            batch_size: 64,
        }
    }
}

pub struct NircTrainer {
    pub network: Box<NircMlp>,
    pub optimizer: Box<AdamOptimizer>,
    pub config: NircConfig,
}

impl NircTrainer {
    pub fn new(config: NircConfig) -> Self {
        Self {
            network: NircMlp::new_boxed(),
            optimizer: AdamOptimizer::new_boxed(config.learning_rate),
            config,
        }
    }

    /// Exécute une passe d'apprentissage (training) en sélectionnant de nouveaux rayons.
    /// Pour chaque étape du "budget", un rayon de training est généré dans la scène,
    /// rebondit via Monte Carlo, et ramène une couleur cible qui sert de vérité terrain.
    /// Les données sont accumulées dans un batch et envoyées au réseau dès que le batch est plein.
    pub fn train(&mut self, tracer: &crate::PathTracer, rng: &mut crate::Pcg32, budget: usize) {
        let mut batch = Vec::with_capacity(self.config.batch_size);

        for _ in 0..budget {
            if let Some((input, target)) = tracer.trace_training_path(rng) {
                let target_vec = nalgebra::SVector::<f32, 3>::new(target.x, target.y, target.z);
                batch.push((input, target_vec));

                if batch.len() >= self.config.batch_size {
                    self.network.train_batch(&mut *self.optimizer, &batch);
                    batch.clear();
                }
            }
        }

        // Applique les gradients restants si le batch n'est pas complètement vide
        if !batch.is_empty() {
            self.network.train_batch(&mut *self.optimizer, &batch);
        }
    }
}

pub const POS_BINS: usize = 8;
pub const DIR_BINS: usize = 8;
pub const INPUT_DIM: usize = (POS_BINS + DIR_BINS) * 3;
const HIDDEN_DIM: usize = 64;
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
#[derive(Clone, Debug)]
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

impl Default for NircMlp {
    fn default() -> Self {
        Self::new()
    }
}

impl NircMlp {
    pub fn new_boxed() -> Box<Self> {
        // Spawn a temporary thread with a large stack to initialize the large SMatrix fields
        // without overflowing the main thread's stack. Returns a heap-allocated Box.
        std::thread::Builder::new()
            .name("NircMlp-Init".into())
            .stack_size(16 * 1024 * 1024)
            .spawn(|| Box::new(Self::new()))
            .expect("Failed to spawn init thread")
            .join()
            .unwrap()
    }

    pub fn new() -> Self {
        let mut rng = crate::Pcg32::from_seed_128(42);
        let mut init_weights = |rows: usize, cols: usize| -> f32 {
            let limit = (6.0 / (rows as f32 + cols as f32)).sqrt();
            (rng.next_f32() * 2.0 - 1.0) * limit
        };

        Self {
            w1: SMatrix::from_fn(|_, _| init_weights(HIDDEN_DIM, INPUT_DIM)),
            b1: SVector::zeros(),
            w2: SMatrix::from_fn(|_, _| init_weights(HIDDEN_DIM, HIDDEN_DIM)),
            b2: SVector::zeros(),
            w3: SMatrix::from_fn(|_, _| init_weights(HIDDEN_DIM, HIDDEN_DIM)),
            b3: SVector::zeros(),
            w4: SMatrix::from_fn(|_, _| init_weights(OUTPUT_DIM, HIDDEN_DIM)),
            b4: SVector::zeros(),
        }
    }

    /// Exponential Moving Average update
    pub fn ema(&mut self, other: &Self, alpha: f32) {
        let beta = 1.0 - alpha;
        self.w1 = self.w1 * beta + other.w1 * alpha;
        self.b1 = self.b1 * beta + other.b1 * alpha;
        self.w2 = self.w2 * beta + other.w2 * alpha;
        self.b2 = self.b2 * beta + other.b2 * alpha;
        self.w3 = self.w3 * beta + other.w3 * alpha;
        self.b3 = self.b3 * beta + other.b3 * alpha;
        self.w4 = self.w4 * beta + other.w4 * alpha;
        self.b4 = self.b4 * beta + other.b4 * alpha;
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

        // Normalize direction to [0, 1]
        let norm_dir = nalgebra::Vector3::new(
            (dir.x + 1.0) * 0.5,
            (dir.y + 1.0) * 0.5,
            (dir.z + 1.0) * 0.5,
        );

        // Oneblob direction encoding
        let dir_sigma = 1.0 / (DIR_BINS as f32);
        let dir_inv_2sigma2 = (DIR_BINS as f32).powi(2) / 2.0;

        for i in 0..DIR_BINS {
            let c = (i as f32 + 0.5) * dir_sigma;
            input[idx] = (-(norm_dir.x - c).powi(2) * dir_inv_2sigma2).exp();
            input[idx + 1] = (-(norm_dir.y - c).powi(2) * dir_inv_2sigma2).exp();
            input[idx + 2] = (-(norm_dir.z - c).powi(2) * dir_inv_2sigma2).exp();
            idx += 3;
        }

        input
    }

    /// Entraîne le réseau sur un batch donné
    pub fn train_batch(
        &mut self,
        opt: &mut AdamOptimizer,
        batch: &[(SVector<f32, INPUT_DIM>, SVector<f32, 3>)],
    ) {
        if batch.is_empty() {
            return;
        }
        let mut accumulated_grads = NircMlpGradients::zeros();
        for (input, target) in batch {
            let (pred, cache) = self.forward_for_training(*input);
            let dl_dy = Self::compute_loss_derivative(&pred, target);
            let grads = self.backward(&cache, dl_dy);
            accumulated_grads.add_gradients(&grads);
        }
        accumulated_grads.divide_by(batch.len() as f32);
        opt.step(self, &accumulated_grads);
    }

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
    ) -> (SVector<f32, OUTPUT_DIM>, NircForwardCache) {
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

        let cache = NircForwardCache {
            x,
            z1,
            a1,
            z2,
            a2,
            z3,
            a3,
        };

        (z4, cache)
    }

    /// Backward pass computing the gradients.
    /// dl_dy is the gradient of the loss with respect to the network's output.
    pub fn backward(
        &self,
        cache: &NircForwardCache,
        dl_dy: SVector<f32, OUTPUT_DIM>,
    ) -> NircMlpGradients {
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

        NircMlpGradients {
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
    pub m: NircMlpGradients,
    pub v: NircMlpGradients,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub lr: f32,
    pub t: u32,
}

impl AdamOptimizer {
    pub fn new_boxed(learning_rate: f32) -> Box<Self> {
        std::thread::Builder::new()
            .name("Adam-Init".into())
            .stack_size(32 * 1024 * 1024)
            .spawn(move || Box::new(Self::new(learning_rate)))
            .expect("Failed to spawn init thread")
            .join()
            .unwrap()
    }

    /// Creates a new Adam optimizer with default parameters.

    pub fn new(learning_rate: f32) -> Self {
        Self {
            m: NircMlpGradients::zeros(),
            v: NircMlpGradients::zeros(),
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            lr: learning_rate,
            t: 0,
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
