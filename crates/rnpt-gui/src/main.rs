mod app;
mod camera_nav;
mod export;
mod panels;
// Shared with the headless renderer; re-exported so `crate::...` paths keep working.
pub(crate) use rnpt_gui::{asset_importer, tonemap};
mod viewport;

use eframe::egui;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Rust Neural Path Tracer")
            .with_inner_size([1100.0, 860.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Rust Neural Path Tracer",
        options,
        Box::new(|cc| {
            cc.egui_ctx.set_visuals(egui::Visuals::dark());
            Ok(Box::new(app::RnptGuiApp::new(cc)))
        }),
    )
}
