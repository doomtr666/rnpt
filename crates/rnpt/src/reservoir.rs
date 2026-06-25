use crate::{Color, Pcg32};
use nalgebra::Point3;

#[derive(Clone)]
pub struct Reservoir {
    /// World-space point on the selected light surface.
    /// For infinite lights (directional / env), sentinel: `wi * 1e20`,
    /// so `normalize(light_pos - any_scene_point) ≈ wi`.
    pub light_pos: Point3<f32>,
    pub li: Color,         // radiance of selected light candidate
    pub w_sum: f32,        // accumulated RIS weight sum over all streamed candidates
    pub m: u32,            // number of candidates streamed (0 = uninitialized)
    /// Stored unbiased contribution weight W from the previous frame.
    /// Next frame's temporal combine: `combine_w = p̂_cur(y_prev) * big_w_stored * m_prev`.
    pub big_w_stored: f32,
    /// Hit position and normal for geometry-aware temporal reuse rejection
    pub hit_pos: Point3<f32>,
    pub hit_normal: nalgebra::Vector3<f32>,
}

impl Default for Reservoir {
    fn default() -> Self {
        Self {
            light_pos: Point3::origin(),
            li: Color::zeros(),
            w_sum: 0.0,
            m: 0,
            big_w_stored: 0.0,
            hit_pos: Point3::origin(),
            hit_normal: nalgebra::Vector3::zeros(),
        }
    }
}

impl Reservoir {
    /// Stream one candidate into the reservoir (Algorithm 2, Bitterli et al. 2020).
    /// `w` is the RIS weight: target_pdf / source_pdf.
    #[inline]
    pub fn update(&mut self, light_pos: Point3<f32>, li: Color, w: f32, rng: &mut Pcg32) {
        self.w_sum += w;
        self.m += 1;
        if rng.next_f32() * self.w_sum < w {
            self.light_pos = light_pos;
            self.li = li;
        }
    }

    #[inline]
    pub fn is_valid(&self) -> bool {
        self.m > 0 && self.w_sum > 0.0
    }

    /// Unbiased contribution weight W = w_sum / (m × p̂).
    #[inline]
    pub fn big_w(&self, p_hat: f32) -> f32 {
        if p_hat > 0.0 {
            self.w_sum / (self.m as f32 * p_hat)
        } else {
            0.0
        }
    }
}
