mod encoding;
mod mlp;
mod optimizer;
mod ring;
mod trainer;

pub use encoding::{encode_inputs, DIR_FREQS, INPUT_DIM, POS_BINS};
pub use mlp::{NircForwardCache, NircMlp, NircMlpGradients};
pub use optimizer::AdamOptimizer;
pub use ring::{NircRingBuffer, RING_CAPACITY};
pub use trainer::{NircConfig, NircTrainer};
