use nalgebra::SVector;

pub const POS_BINS: usize = 1;
/// Number of sinusoidal NeRF frequencies applied to each direction component.
/// Frequency k has period 2/2^k in [-1,1]; at k=9 the period is ~0.004 in dz
/// → angular resolution ~3.6° even near the poles of the unit sphere.
pub const DIR_FREQS: usize = 1;
pub const INPUT_DIM: usize = POS_BINS * 3 + DIR_FREQS * 2 * 3;
pub(super) const HIDDEN_DIM: usize = 16;
pub(super) const OUTPUT_DIM: usize = 3;

/// NeRF positional encoding for position (oneblob) + direction (sinusoidal).
/// Returns the INPUT_DIM-dimensional feature vector fed to the MLP.
pub fn encode_inputs(
    pos: &nalgebra::Point3<f32>,
    dir: &nalgebra::Vector3<f32>,
    bounds_min: &nalgebra::Point3<f32>,
    bounds_max: &nalgebra::Point3<f32>,
) -> SVector<f32, INPUT_DIM> {
    let mut input = SVector::<f32, INPUT_DIM>::zeros();
    let mut idx = 0;

    let extents = bounds_max - bounds_min;
    let norm_pos = nalgebra::Point3::new(
        if extents.x > 0.0 { (pos.x - bounds_min.x) / extents.x } else { 0.5 },
        if extents.y > 0.0 { (pos.y - bounds_min.y) / extents.y } else { 0.5 },
        if extents.z > 0.0 { (pos.z - bounds_min.z) / extents.z } else { 0.5 },
    );

    // Oneblob position encoding.
    let pos_sigma = 1.0 / (POS_BINS as f32);
    let pos_inv_2sigma2 = (POS_BINS as f32).powi(2) / 2.0;
    for i in 0..POS_BINS {
        let c = (i as f32 + 0.5) * pos_sigma;
        input[idx]     = (-(norm_pos.x - c).powi(2) * pos_inv_2sigma2).exp();
        input[idx + 1] = (-(norm_pos.y - c).powi(2) * pos_inv_2sigma2).exp();
        input[idx + 2] = (-(norm_pos.z - c).powi(2) * pos_inv_2sigma2).exp();
        idx += 3;
    }

    // NeRF sinusoidal direction encoding.
    // dir components are in [-1, 1] (unit vector). Each frequency k encodes
    // sin(2^k π d) and cos(2^k π d) per component → 2×3 features per k.
    for k in 0..DIR_FREQS {
        let freq = (1u32 << k) as f32 * std::f32::consts::PI;
        input[idx]     = (freq * dir.x).sin();
        input[idx + 1] = (freq * dir.y).sin();
        input[idx + 2] = (freq * dir.z).sin();
        idx += 3;
        input[idx]     = (freq * dir.x).cos();
        input[idx + 1] = (freq * dir.y).cos();
        input[idx + 2] = (freq * dir.z).cos();
        idx += 3;
    }

    input
}
