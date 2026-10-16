//! Everything needed to reproduce a render: its settings (saved as JSON next to
//! every image), building a tracer config from them, and writing the outputs.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::tonemap::{tonemap_and_convert, TonemapOperator};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CameraSettings {
    pub position: [f32; 3],
    pub target: [f32; 3],
    pub fov: f32,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct EnvSettings {
    pub path: String,
    pub intensity: f32,
    pub rotation_deg: f32,
}

/// Everything that defines a render. Paths are relative to the directory the
/// GUI or `rnpt-render` runs from (the repository root, next to `assets/`).
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RenderSettings {
    pub scene: String,
    pub camera: CameraSettings,
    pub width: usize,
    pub height: usize,
    pub strategy: String,
    pub rays_per_pixel: u64,
    pub samples_per_pixel: u32,
    pub env: Option<EnvSettings>,
    pub exposure: f32,
    pub tonemapper: String,
    /// Absent from settings saved before the option existed.
    #[serde(default)]
    pub indirect_only: bool,
}

/// What a finished render measured, saved alongside its settings.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct RenderStats {
    pub rays_per_pixel: f64,
    pub samples_per_pixel: f64,
    pub seconds: f64,
    /// Only known to the headless renderer, which owns the whole run.
    pub paths: Option<u64>,
    pub rays: Option<u64>,
    pub threads: Option<usize>,
    pub nirc_steps: Option<u64>,
    pub train_s: Option<f64>,
    pub pacing_wait_s: Option<f64>,
}

#[derive(Serialize, Deserialize)]
struct SavedRender {
    settings: RenderSettings,
    stats: RenderStats,
}

pub fn strategy_name(s: rnpt::SamplingStrategy) -> &'static str {
    match s {
        rnpt::SamplingStrategy::Nirc => "nirc",
        rnpt::SamplingStrategy::Mis => "mis",
        rnpt::SamplingStrategy::NeeOnly => "neeonly",
        rnpt::SamplingStrategy::BrdfOnly => "brdfonly",
        rnpt::SamplingStrategy::DirectOnly => "directonly",
    }
}

pub fn parse_strategy(name: &str) -> Result<rnpt::SamplingStrategy, String> {
    Ok(match name {
        "nirc" => rnpt::SamplingStrategy::Nirc,
        "mis" => rnpt::SamplingStrategy::Mis,
        "neeonly" | "nee" => rnpt::SamplingStrategy::NeeOnly,
        "brdfonly" | "brdf" => rnpt::SamplingStrategy::BrdfOnly,
        "directonly" | "direct" => rnpt::SamplingStrategy::DirectOnly,
        _ => return Err(format!("unknown strategy {name} (nirc, mis, nee, brdf, direct)")),
    })
}

pub fn tonemapper_name(t: TonemapOperator) -> &'static str {
    match t {
        TonemapOperator::Aces => "aces",
        TonemapOperator::Reinhard => "reinhard",
    }
}

pub fn parse_tonemapper(name: &str) -> TonemapOperator {
    if name == "reinhard" { TonemapOperator::Reinhard } else { TonemapOperator::Aces }
}

/// Append the environment light, if any, to the light list. Returns the new
/// list and the index of the environment in it.
pub fn with_env(
    lights: &Arc<Vec<rnpt::Light>>,
    env: &Option<Arc<rnpt::EnvLight>>,
) -> (Arc<Vec<rnpt::Light>>, Option<usize>) {
    match env {
        Some(e) => {
            let mut v = (**lights).clone();
            v.push(rnpt::Light::Environment(e.clone()));
            let idx = v.len() - 1;
            (Arc::new(v), Some(idx))
        }
        None => (lights.clone(), None),
    }
}

/// Build the BVH and the light list of a scene. Returns them with the build
/// time in ms and the triangle count.
pub fn build_bvh_and_lights(scene: &rnpt::Scene) -> (Arc<rnpt::Bvh>, Arc<Vec<rnpt::Light>>, u64, usize) {
    let t0 = Instant::now();
    let (bvh, emitters) = rnpt::BvhBuilder::new(scene).build();
    let build_ms = t0.elapsed().as_millis() as u64;
    let tri_count = bvh.triangle_meta.len();
    let lights = rnpt::build_lights(&scene.lights, emitters);
    (Arc::new(bvh), Arc::new(lights), build_ms, tri_count)
}

/// Load the scene and environment of `settings` into a tracer config.
pub fn load_config(settings: &RenderSettings) -> Result<rnpt::PathTracerConfig, String> {
    let (scene, _) =
        crate::asset_importer::import_scene(&settings.scene).map_err(|e| format!("{}: {e}", settings.scene))?;
    let scene = Arc::new(scene);
    let (bvh, lights, _, _) = build_bvh_and_lights(&scene);

    let env = match &settings.env {
        Some(e) => {
            let (pixels, w, h) =
                crate::asset_importer::load_hdr(Path::new(&e.path)).ok_or(format!("{}: cannot load", e.path))?;
            Some(Arc::new(rnpt::EnvLight::new(pixels, w, h, e.intensity, e.rotation_deg.to_radians())))
        }
        None => None,
    };
    let (lights, env) = with_env(&lights, &env);

    let c = &settings.camera;
    Ok(rnpt::PathTracerConfig {
        width: settings.width,
        height: settings.height,
        camera: rnpt::Camera {
            position: rnpt::Point3::new(c.position[0], c.position[1], c.position[2]),
            target: rnpt::Point3::new(c.target[0], c.target[1], c.target[2]),
            fov: c.fov,
        },
        scene,
        bvh,
        lights,
        env,
        strategy: parse_strategy(&settings.strategy)?,
        nirc_network: None,
        budget: rnpt::RenderBudget {
            rays_per_pixel: settings.rays_per_pixel,
            samples_per_pixel: settings.samples_per_pixel,
        },
        indirect_only: settings.indirect_only,
    })
}

/// Average rays and samples per pixel actually spent.
pub fn pixel_averages(pixels: &[rnpt::Pixel]) -> (f64, f64) {
    let n = pixels.len().max(1) as f64;
    let rays = pixels.iter().map(|p| p.rays).sum::<u64>() as f64 / n;
    let spp = pixels.iter().map(|p| p.samples as u64).sum::<u64>() as f64 / n;
    (rays, spp)
}

/// Write `<dir>/<name>.png` (as displayed), `.hdr` (linear radiance) and `.json`
/// (settings and stats, enough to re-render it with `rnpt-render --settings`).
/// The name records how the image was made. Returns the PNG path.
pub fn save_render(
    dir: &Path,
    settings: &RenderSettings,
    stats: &RenderStats,
    pixels: &[rnpt::Pixel],
) -> Result<PathBuf, String> {
    let (w, h) = (settings.width, settings.height);
    if pixels.len() != w * h || pixels.is_empty() {
        return Err("nothing rendered yet".into());
    }
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;

    let scene = Path::new(&settings.scene).file_stem().map_or("scene".into(), |s| s.to_string_lossy().into_owned());
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs());
    let indirect = if settings.indirect_only { "-indirect" } else { "" };
    let stem = format!(
        "{scene}_{}{indirect}_{:.0}rpp_{:.1}spp_{:.2}s_{stamp}",
        settings.strategy, stats.rays_per_pixel, stats.samples_per_pixel, stats.seconds
    );
    // The stem contains dots (e.g. "12.34s"): append extensions, never with_extension.
    let path = |ext: &str| dir.join(format!("{stem}{ext}"));

    let mut rgba = vec![0u8; w * h * 4];
    tonemap_and_convert(pixels, settings.exposure, parse_tonemapper(&settings.tonemapper), &mut rgba);
    let png = path(".png");
    image::save_buffer(&png, &rgba, w as u32, h as u32, image::ExtendedColorType::Rgba8)
        .map_err(|e| format!("{}: {e}", png.display()))?;

    let radiance: Vec<image::Rgb<f32>> = pixels
        .iter()
        .map(|p| {
            let c = p.accumulated_radiance;
            image::Rgb(if p.samples == 0 { [0.0; 3] } else { [c.x, c.y, c.z] })
        })
        .collect();
    let hdr = path(".hdr");
    let file = std::fs::File::create(&hdr).map_err(|e| format!("{}: {e}", hdr.display()))?;
    image::codecs::hdr::HdrEncoder::new(std::io::BufWriter::new(file))
        .encode(&radiance, w, h)
        .map_err(|e| format!("{}: {e}", hdr.display()))?;

    let json = path(".json");
    let saved = SavedRender { settings: settings.clone(), stats: stats.clone() };
    std::fs::write(&json, serde_json::to_string_pretty(&saved).unwrap()).map_err(|e| format!("{}: {e}", json.display()))?;

    Ok(png)
}

/// Read the settings saved next to a render (`.json`).
pub fn load_settings(path: &Path) -> Result<RenderSettings, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let saved: SavedRender = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(saved.settings)
}
