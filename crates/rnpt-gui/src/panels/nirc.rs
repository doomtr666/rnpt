use eframe::egui;
use crate::app::RnptGuiApp;

pub fn show(app: &mut RnptGuiApp, ui: &mut egui::Ui) {
    ui.separator();
    ui.add_space(4.0);

    let ring = app.tracer.as_ref().map_or(0, |t| t.ring_filled());
    ui.label(format!("Ring: {} samples", ring));
    if app.loss_ema > 0.0 {
        ui.label(format!("Loss (ema): {:.5}", app.loss_ema));
        ui.label(format!("Loss (raw): {:.5}", app.last_loss));
    }
    if app.rel_error > 0.0 {
        ui.label(format!("RelMean: {:.2}%", app.rel_error * 100.0));
    }

    ui.add_space(4.0);
    ui.collapsing("Directional probe", |ui| {
        if let Some(ref probe_tex) = app.probe_texture {
            ui.image(egui::load::SizedTexture::new(
                probe_tex.id(),
                egui::vec2(256.0, 128.0),
            ));
            ui.label("← azimuth 0→360°  |  elevation: top→bottom ↓");
        } else {
            ui.label("Ctrl+click on the image to place a probe");
        }
    });
}
