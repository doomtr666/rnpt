use nalgebra::SVector;
use super::encoding::{INPUT_DIM, OUTPUT_DIM};
use super::mlp::{NircMlp, NircMlpGradients};
use super::optimizer::AdamOptimizer;

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
    /// EMA-smoothed inference snapshot, updated after each training step.
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

    /// Train on pre-collected samples in mini-batches. Returns average loss.
    /// Updates `ema_buf` in-place after all batches (no allocation).
    pub fn train_samples(
        &mut self,
        samples: &[(SVector<f32, INPUT_DIM>, SVector<f32, OUTPUT_DIM>)],
    ) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let mut total_loss = 0.0f32;
        let mut num_batches = 0usize;
        for chunk in samples.chunks(self.config.batch_size) {
            total_loss += self.train_one_batch(chunk);
            num_batches += 1;
        }
        self.ema_buf.ema(&self.network, self.config.ema_alpha);
        if num_batches > 0 { total_loss / num_batches as f32 } else { 0.0 }
    }

    fn train_one_batch(
        &mut self,
        batch: &[(SVector<f32, INPUT_DIM>, SVector<f32, OUTPUT_DIM>)],
    ) -> f32 {
        if batch.is_empty() {
            return 0.0;
        }
        self.grads.zero_in_place();
        let mut batch_loss = 0.0f32;
        for (input, target) in batch {
            let (pred, cache) = self.network.forward_for_training(*input);
            let dl_dy = NircMlp::compute_loss_derivative(&pred, target);
            batch_loss += NircMlp::compute_loss(&pred, target);
            self.network.backward_into(&cache, dl_dy, &mut self.grads);
        }
        self.grads.divide_by(batch.len() as f32);
        self.optimizer.step(&mut self.network, &self.grads);
        batch_loss / batch.len() as f32
    }
}
