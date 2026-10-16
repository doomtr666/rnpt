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
        // A ball resting on the probed surface, seen from the camera: each point
        // shows what the network predicts arriving from the direction it faces.
        const SIZE: f32 = 192.0;
        let response = ui.image(egui::load::SizedTexture::new(probe_tex.id(), egui::vec2(SIZE, SIZE)));
        let rect = response.rect;
        let to_screen = |uv: [f32; 2]| rect.center() + egui::vec2(uv[0], -uv[1]) * (SIZE * 0.5);
        ui.painter().circle_stroke(rect.center(), SIZE * 0.5, egui::Stroke::new(1.0, egui::Color32::GRAY));
        if let Some(nrm) = app.probe_normal {
            let p = to_screen(nrm);
            let stroke = egui::Stroke::new(1.5, egui::Color32::from_rgb(255, 200, 0));
            ui.painter().line_segment([p - egui::vec2(5.0, 0.0), p + egui::vec2(5.0, 0.0)], stroke);
            ui.painter().line_segment([p - egui::vec2(0.0, 5.0), p + egui::vec2(0.0, 5.0)], stroke);
        }
        ui.label(format!("Training steps: {}", app.probe_steps));
        ui.add(egui::Slider::new(&mut app.probe_gain, 0.1..=20.0).logarithmic(true).text("Gain"));
        ui.checkbox(&mut app.probe_back, "Back side (other half of the ball)");
        ui.label(
            "A ball on the surface, seen from the camera, lit by the\n\
             network's prediction. Gray: below the surface.\n\
             Yellow cross: surface normal.",
        );
        if ui.button("Remove probe").clicked() {
            app.probe_pixel = None;
            app.probe_texture = None;
        }
    });
}
