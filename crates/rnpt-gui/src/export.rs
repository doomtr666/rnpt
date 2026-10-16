//! "Save image": the current render with its settings, so that it can be
//! re-rendered headless (`rnpt-render --settings <render>.json`).

use std::path::{Path, PathBuf};

use rnpt_gui::render_io::{self, CameraSettings, EnvSettings, RenderSettings, RenderStats};

use crate::app::RnptGuiApp;

const RENDER_DIR: &str = "renders";

pub fn current_settings(app: &RnptGuiApp) -> RenderSettings {
    let scene = app.asset_files.get(app.selected_asset_index).map_or(String::new(), |p| p.to_string_lossy().into_owned());
    let env = (app.selected_env > 0).then(|| EnvSettings {
        path: app.hdr_files[app.selected_env - 1].to_string_lossy().into_owned(),
        intensity: app.env_intensity,
        rotation_deg: app.env_rotation,
    });
    let (p, t) = (app.camera.position, app.camera.target);
    RenderSettings {
        scene,
        camera: CameraSettings { position: [p.x, p.y, p.z], target: [t.x, t.y, t.z], fov: app.camera.fov },
        width: app.local_width,
        height: app.local_height,
        strategy: render_io::strategy_name(app.strategy).into(),
        rays_per_pixel: app.budget.rays_per_pixel,
        samples_per_pixel: app.budget.samples_per_pixel,
        env,
        exposure: app.exposure,
        tonemapper: render_io::tonemapper_name(app.tonemapper).into(),
        indirect_only: app.indirect_only,
    }
}

pub fn save_render(app: &RnptGuiApp) -> Result<PathBuf, String> {
    let (rays_per_pixel, samples_per_pixel) = render_io::pixel_averages(&app.local_pixels);
    let stats = RenderStats {
        rays_per_pixel,
        samples_per_pixel,
        seconds: app.render_time.unwrap_or_else(|| app.render_start.elapsed()).as_secs_f64(),
        ..Default::default()
    };
    render_io::save_render(Path::new(RENDER_DIR), &current_settings(app), &stats, &app.local_pixels)
}
