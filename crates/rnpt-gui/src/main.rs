mod app;
mod asset_importer;
mod camera_nav;
mod panels;
mod tonemap;
mod viewport;

use eframe::egui;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Rust Neural Path Tracer")
            .with_inner_size([1100.0, 700.0]),
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
