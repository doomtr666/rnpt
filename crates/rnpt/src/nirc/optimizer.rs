use super::mlp::{NircMlp, NircMlpGradients};

/// Adam optimizer state: first moment (m) and second moment (v) for every parameter.
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
    /// Allocate on the heap — `AdamOptimizer` contains two `NircMlpGradients` (~1.24 MB
    /// total); `alloc_zeroed` avoids constructing them on the caller's stack.
    pub fn new_boxed(learning_rate: f32) -> Box<Self> {
        unsafe {
            let layout = std::alloc::Layout::new::<Self>();
            let ptr = std::alloc::alloc_zeroed(layout) as *mut Self;
            if ptr.is_null() { std::alloc::handle_alloc_error(layout); }
            let b = &mut *ptr;
            // m and v are zeroed by alloc_zeroed — correct initial state.
            b.beta1   = 0.9;
            b.beta2   = 0.999;
            b.epsilon = 1e-8;
            b.lr      = learning_rate;
            b.t       = 0;
            Box::from_raw(ptr)
        }
    }

    /// Apply accumulated gradients to the network weights via the Adam update rule.
    pub fn step(&mut self, network: &mut NircMlp, grads: &NircMlpGradients) {
        self.t += 1;

        let b1_t = 1.0 - self.beta1.powi(self.t as i32);
        let b2_t = 1.0 - self.beta2.powi(self.t as i32);
        let lr_t = self.lr * (b2_t.sqrt() / b1_t);

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
