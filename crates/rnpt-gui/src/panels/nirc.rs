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

    ui.add_space(4.0);
    ui.collapsing("Directional probe", |ui| {
        let Some(ref probe_tex) = app.probe_texture else {
            ui.label("Ctrl+click on the image to place a probe");
            return;
        };
        // What the network predicts arriving at the probed point, over the
        // hemisphere above the surface, updated live as it trains.
        const SIZE: f32 = 192.0;
        let response = ui.image(egui::load::SizedTexture::new(probe_tex.id(), egui::vec2(SIZE, SIZE)));
        let rect = response.rect;
        let painter = ui.painter();
        let to_screen = |uv: [f32; 2]| rect.center() + egui::vec2(uv[0], -uv[1]) * (SIZE * 0.5);
        painter.circle_stroke(rect.center(), SIZE * 0.5, egui::Stroke::new(1.0, egui::Color32::GRAY));
        painter.text(
            to_screen([0.0, 1.0]) + egui::vec2(0.0, -2.0),
            egui::Align2::CENTER_BOTTOM,
            "up",
            egui::FontId::proportional(11.0),
            egui::Color32::GRAY,
        );
        if let Some(cam) = app.probe_camera {
            painter.circle_stroke(to_screen(cam), 4.0, egui::Stroke::new(1.5, egui::Color32::from_rgb(255, 200, 0)));
        }
        ui.add_space(14.0);
        ui.label(format!("Training steps: {}", app.probe_steps));
        ui.add(egui::Slider::new(&mut app.probe_gain, 0.1..=20.0).logarithmic(true).text("Gain"));
        ui.label("Center: surface normal. Rim: horizon. Equal-area disk.\nYellow ring: toward the camera.");
        if ui.button("Remove probe").clicked() {
            app.probe_pixel = None;
            app.probe_texture = None;
        }
    });
}
