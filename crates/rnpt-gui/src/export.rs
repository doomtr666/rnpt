//! Save the current render: a PNG of what is on screen, and a linear Radiance
//! .hdr of the raw radiance for numerical comparisons (MSE against a reference).

use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::app::RnptGuiApp;
use crate::tonemap::tonemap_and_convert;

const RENDER_DIR: &str = "renders";

/// Write `renders/<name>.png` and `renders/<name>.hdr`, the name recording how
/// the image was made. Returns the PNG path.
pub fn save_render(app: &RnptGuiApp) -> Result<PathBuf, String> {
    let (w, h) = (app.local_width, app.local_height);
    let pixels = &app.local_pixels;
    if pixels.is_empty() {
        return Err("nothing rendered yet".into());
    }

    std::fs::create_dir_all(RENDER_DIR).map_err(|e| format!("{RENDER_DIR}: {e}"))?;
    // The stem contains dots (e.g. "12.34s"): append extensions, never with_extension.
    let stem = PathBuf::from(RENDER_DIR).join(file_stem(app)).into_os_string();
    let with_ext = |ext: &str| { let mut p = stem.clone(); p.push(ext); PathBuf::from(p) };

    let mut rgba = vec![0u8; w * h * 4];
    tonemap_and_convert(pixels, app.exposure, app.tonemapper, &mut rgba);
    let png = with_ext(".png");
    image::save_buffer(&png, &rgba, w as u32, h as u32, image::ExtendedColorType::Rgba8)
        .map_err(|e| format!("{}: {e}", png.display()))?;

    let radiance: Vec<image::Rgb<f32>> = pixels
        .iter()
        .map(|p| {
            let c = if p.samples == 0 { [0.0; 3] } else { [p.accumulated_radiance.x, p.accumulated_radiance.y, p.accumulated_radiance.z] };
            image::Rgb(c)
        })
        .collect();
    let hdr = with_ext(".hdr");
    let file = std::fs::File::create(&hdr).map_err(|e| format!("{}: {e}", hdr.display()))?;
    image::codecs::hdr::HdrEncoder::new(std::io::BufWriter::new(file))
        .encode(&radiance, w, h)
        .map_err(|e| format!("{}: {e}", hdr.display()))?;

    Ok(png)
}

/// e.g. `bistro_ext_nirc_512rpp_38.2spp_12.34s_1792412345`
fn file_stem(app: &RnptGuiApp) -> String {
    let scene = app
        .asset_files
        .get(app.selected_asset_index)
        .and_then(|p| p.file_stem())
        .map_or("scene".into(), |s| s.to_string_lossy().into_owned());
    let strategy = format!("{:?}", app.strategy).to_lowercase();

    let n = app.local_pixels.len() as f64;
    let rpp = app.local_pixels.iter().map(|p| p.rays).sum::<u64>() as f64 / n;
    let spp = app.local_pixels.iter().map(|p| p.samples as u64).sum::<u64>() as f64 / n;
    let time = app.render_time.unwrap_or_else(|| app.render_start.elapsed()).as_secs_f64();
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs());

    format!("{scene}_{strategy}_{rpp:.0}rpp_{spp:.1}spp_{time:.2}s_{stamp}")
}
