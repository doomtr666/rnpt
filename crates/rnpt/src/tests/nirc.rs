use crate::nirc::{
    encode_inputs, INPUT_DIM, NircMlp, NircRingBuffer, RING_CAPACITY,
};
use crate::Pcg32;
use nalgebra::{Point3, SVector, Vector3};

// ── encode_inputs ─────────────────────────────────────────────────────────────

fn make_bounds() -> (Point3<f32>, Point3<f32>) {
    (Point3::new(-5.0, -2.0, -5.0), Point3::new(5.0, 4.0, 5.0))
}

#[test]
fn encode_inputs_deterministic() {
    let pos = Point3::new(1.0, 0.5, -1.0);
    let dir = Vector3::new(0.0, 1.0, 0.0);
    let (bmin, bmax) = make_bounds();
    let a = encode_inputs(&pos, &dir, &bmin, &bmax);
    let b = encode_inputs(&pos, &dir, &bmin, &bmax);
    assert_eq!(a, b);
}

#[test]
fn encode_inputs_output_dim() {
    let pos = Point3::origin();
    let dir = Vector3::new(0.0, 0.0, 1.0);
    let (bmin, bmax) = make_bounds();
    let v = encode_inputs(&pos, &dir, &bmin, &bmax);
    assert_eq!(v.len(), INPUT_DIM);
}

#[test]
fn encode_inputs_different_positions_give_different_outputs() {
    let (bmin, bmax) = make_bounds();
    let dir = Vector3::new(0.0, 1.0, 0.0);
    let a = encode_inputs(&Point3::new(0.0, 0.0, 0.0), &dir, &bmin, &bmax);
    let b = encode_inputs(&Point3::new(4.0, 2.0, 3.0), &dir, &bmin, &bmax);
    assert_ne!(a, b, "different positions should produce different encodings");
}

#[test]
fn encode_inputs_different_directions_give_different_outputs() {
    let (bmin, bmax) = make_bounds();
    let pos = Point3::new(1.0, 1.0, 1.0);
    let a = encode_inputs(&pos, &Vector3::new(1.0, 0.0, 0.0), &bmin, &bmax);
    let b = encode_inputs(&pos, &Vector3::new(0.0, 1.0, 0.0), &bmin, &bmax);
    assert_ne!(a, b, "different directions should produce different encodings");
}

#[test]
fn encode_inputs_position_at_bounds_min() {
    // Position exactly at bounds_min → normalized pos = (0,0,0).
    let bmin = Point3::new(-1.0, -1.0, -1.0);
    let bmax = Point3::new(1.0, 1.0, 1.0);
    let dir  = Vector3::new(0.0, 1.0, 0.0);
    let v = encode_inputs(&bmin, &dir, &bmin, &bmax);
    for &x in v.iter() {
        assert!(x.is_finite(), "non-finite encoding: {x}");
    }
}

// ── NircMlp loss ─────────────────────────────────────────────────────────────

#[test]
fn loss_zero_when_pred_equals_target() {
    let x: SVector<f32, 3> = SVector::from([1.0, 2.0, 0.5]);
    let loss = NircMlp::compute_loss(&x, &x);
    assert!(loss.abs() < 1e-6, "loss should be 0 when pred == target: {loss}");
}

#[test]
fn loss_non_negative() {
    let pred:   SVector<f32, 3> = SVector::from([0.5, 1.0, 2.0]);
    let target: SVector<f32, 3> = SVector::from([1.0, 0.5, 3.0]);
    assert!(NircMlp::compute_loss(&pred, &target) >= 0.0);
}

#[test]
fn loss_symmetric_in_log_space() {
    // log-MSE: (log(1+a) - log(1+b))² = (log(1+b) - log(1+a))²
    let a: SVector<f32, 3> = SVector::from([1.0, 2.0, 3.0]);
    let b: SVector<f32, 3> = SVector::from([2.0, 1.0, 5.0]);
    // Note: the network clips pred to 0 on the negative side, so symmetry only
    // holds when both pred and target are positive.
    let lab = NircMlp::compute_loss(&a, &b);
    let lba = NircMlp::compute_loss(&b, &a);
    assert!((lab - lba).abs() < 1e-5, "loss not symmetric: {lab} vs {lba}");
}

#[test]
fn loss_derivative_zero_at_minimum() {
    let x: SVector<f32, 3> = SVector::from([1.0, 2.0, 0.5]);
    let dl_dy = NircMlp::compute_loss_derivative(&x, &x);
    for &g in dl_dy.iter() {
        assert!(g.abs() < 1e-5, "gradient non-zero at minimum: {g}");
    }
}

#[test]
fn forward_pass_finite() {
    let network = NircMlp::new_boxed();
    let (bmin, bmax) = make_bounds();
    let pos = Point3::new(0.5, 1.0, -0.5);
    let dir = Vector3::new(0.0, 1.0, 0.0);
    let input = encode_inputs(&pos, &dir, &bmin, &bmax);
    let output = network.forward(input);
    for &v in output.iter() {
        assert!(v.is_finite(), "non-finite network output: {v}");
    }
}

// ── NircRingBuffer ────────────────────────────────────────────────────────────

#[test]
fn ring_starts_empty() {
    let ring = NircRingBuffer::new();
    assert_eq!(ring.filled(), 0);
    assert!(ring.read_random_batch(10).is_empty());
}

#[test]
fn ring_filled_increases_with_pushes() {
    let ring = NircRingBuffer::new();
    let input  = SVector::<f32, INPUT_DIM>::zeros();
    let target = SVector::<f32, 3>::zeros();
    ring.push(input, target);
    assert_eq!(ring.filled(), 1);
    ring.push(input, target);
    assert_eq!(ring.filled(), 2);
}

#[test]
fn ring_filled_caps_at_capacity() {
    let ring = NircRingBuffer::new();
    let input  = SVector::<f32, INPUT_DIM>::zeros();
    let target = SVector::<f32, 3>::zeros();
    for _ in 0..(RING_CAPACITY + 10) {
        ring.push(input, target);
    }
    assert_eq!(ring.filled(), RING_CAPACITY);
}

#[test]
fn ring_read_batch_respects_count() {
    let ring = NircRingBuffer::new();
    let input  = SVector::<f32, INPUT_DIM>::zeros();
    let target = SVector::<f32, 3>::zeros();
    for _ in 0..100 {
        ring.push(input, target);
    }
    let batch = ring.read_random_batch(32);
    assert_eq!(batch.len(), 32);
}

#[test]
fn ring_read_batch_capped_by_filled() {
    let ring = NircRingBuffer::new();
    let input  = SVector::<f32, INPUT_DIM>::zeros();
    let target = SVector::<f32, 3>::zeros();
    for _ in 0..5 {
        ring.push(input, target);
    }
    let batch = ring.read_random_batch(100);
    assert_eq!(batch.len(), 5);
}

// ── NircTrainer round-trip ────────────────────────────────────────────────────

#[test]
fn trainer_loss_decreases_on_repeated_training() {
    use crate::nirc::{NircConfig, NircTrainer};

    let config = NircConfig { learning_rate: 1e-3, batch_size: 32, ema_alpha: 0.1 };
    let mut trainer = NircTrainer::new(config);
    let (bmin, bmax) = make_bounds();

    // Build a fixed synthetic dataset: network should learn constant radiance.
    let target_rgb = [2.0f32, 1.0, 0.5];
    let mut rng = Pcg32::from_seed_128(42);
    let samples: Vec<_> = (0..256).map(|_| {
        let pos = Point3::new(
            rng.next_f32() * 10.0 - 5.0,
            rng.next_f32() * 6.0 - 2.0,
            rng.next_f32() * 10.0 - 5.0,
        );
        let theta = rng.next_f32() * std::f32::consts::PI;
        let phi   = rng.next_f32() * 2.0 * std::f32::consts::PI;
        let dir   = Vector3::new(theta.sin() * phi.cos(), theta.cos(), theta.sin() * phi.sin());
        let input  = encode_inputs(&pos, &dir, &bmin, &bmax);
        let target: SVector<f32, 3> = SVector::from(target_rgb);
        (input, target)
    }).collect();

    let loss_before = trainer.train_samples(&samples);
    let mut loss_after = loss_before;
    for _ in 0..20 {
        loss_after = trainer.train_samples(&samples);
    }
    assert!(loss_after < loss_before, "loss did not decrease: {loss_before} → {loss_after}");
}

// ── Micro-benchmark (cargo test --release -p rnpt nirc_inference_cost -- --ignored --nocapture) ──

/// Cost of one NIRC query: encoding, network forward, and the scalar `exp`
/// behind every SiLU activation (48 per forward).
#[test]
#[ignore]
fn nirc_inference_cost() {
    use std::hint::black_box;
    use std::time::Instant;

    const N: usize = 2_000_000;
    let net = NircMlp::new_boxed();
    let (bmin, bmax) = make_bounds();
    let mut rng = Pcg32::from_seed(7, 1);
    let inputs: Vec<(Point3<f32>, Vector3<f32>)> = (0..1024)
        .map(|_| {
            let p = Point3::new(rng.next_f32() * 8.0 - 4.0, rng.next_f32() * 5.0 - 1.5, rng.next_f32() * 8.0 - 4.0);
            let d = Vector3::new(rng.next_f32() - 0.5, rng.next_f32() - 0.5, rng.next_f32() - 0.5).normalize();
            (p, d)
        })
        .collect();
    let encoded: Vec<SVector<f32, INPUT_DIM>> = inputs.iter().map(|(p, d)| encode_inputs(p, d, &bmin, &bmax)).collect();

    let time = |name: &str, f: &dyn Fn(usize) -> f32| {
        let t = Instant::now();
        let mut acc = 0.0f32;
        for i in 0..N {
            acc += f(i);
        }
        black_box(acc);
        println!("{name:<12} {:6.1} ns/call", t.elapsed().as_nanos() as f64 / N as f64);
    };
    time("encode", &|i| {
        let (p, d) = &inputs[i & 1023];
        encode_inputs(black_box(p), black_box(d), &bmin, &bmax)[0]
    });
    time("forward", &|i| net.forward(black_box(encoded[i & 1023]))[0]);
    time("48 x exp", &|i| {
        let x = black_box(i as f32 * 1e-6);
        (0..48).map(|k| (-(x + k as f32 * 0.01)).exp()).sum()
    });
}
