#[derive(Clone, Copy, PartialEq)]
pub enum TonemapOperator {
    Reinhard,
    Aces,
}

pub fn tonemap_and_convert(
    pixels: &[rnpt::Pixel],
    exposure: f32,
    operator: TonemapOperator,
    output_rgba: &mut [u8],
) {
    output_rgba
        .chunks_exact_mut(4)
        .zip(pixels.iter())
        .for_each(|(rgba, pixel)| {
            if pixel.samples == 0 {
                rgba[0] = 0;
                rgba[1] = 0;
                rgba[2] = 0;
                rgba[3] = 255;
                return;
            }

            let r = pixel.accumulated_radiance[0] * exposure;
            let g = pixel.accumulated_radiance[1] * exposure;
            let b = pixel.accumulated_radiance[2] * exposure;

            let (r, g, b) = match operator {
                TonemapOperator::Reinhard => (r / (r + 1.0), g / (g + 1.0), b / (b + 1.0)),
                TonemapOperator::Aces => {
                    // Narkowicz ACES fit
                    let (a, b2, c, d, e) = (2.51f32, 0.03f32, 2.43f32, 0.59f32, 0.14f32);
                    let aces = |v: f32| (v * (a * v + b2)) / (v * (c * v + d) + e);
                    (aces(r), aces(g), aces(b))
                }
            };

            rgba[0] = (r.powf(1.0 / 2.2).clamp(0.0, 1.0) * 255.0) as u8;
            rgba[1] = (g.powf(1.0 / 2.2).clamp(0.0, 1.0) * 255.0) as u8;
            rgba[2] = (b.powf(1.0 / 2.2).clamp(0.0, 1.0) * 255.0) as u8;
            rgba[3] = 255;
        });
}

/// Format path throughput for display (one path = camera ray + all bounces + shadow rays).
pub fn format_paths_per_sec(paths_per_sec: f64) -> String {
    if paths_per_sec >= 1_000_000.0 {
        format!("{:.2} Mpaths/s", paths_per_sec / 1_000_000.0)
    } else if paths_per_sec >= 1_000.0 {
        format!("{:.1} Kpaths/s", paths_per_sec / 1_000.0)
    } else {
        format!("{:.0} paths/s", paths_per_sec)
    }
}

/// Format raw BVH ray throughput for display.
pub fn format_rays_per_sec(rays_per_sec: f64) -> String {
    if rays_per_sec >= 1_000_000.0 {
        format!("{:.2} Mrays/s", rays_per_sec / 1_000_000.0)
    } else if rays_per_sec >= 1_000.0 {
        format!("{:.1} Krays/s", rays_per_sec / 1_000.0)
    } else {
        format!("{:.0} rays/s", rays_per_sec)
    }
}
