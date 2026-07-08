use eframe::egui;
use crate::app::RnptGuiApp;
use crate::tonemap::TonemapOperator;

pub fn show(app: &mut RnptGuiApp, ui: &mut egui::Ui) {
    ui.heading("Post-Processing");
    ui.add_space(4.0);
    ui.add(egui::Slider::new(&mut app.exposure, 0.1..=10.0).text("Exposure"));
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        ui.label("Tonemapper:");
        egui::ComboBox::from_id_source("tonemapper_selector")
            .selected_text(match app.tonemapper {
                TonemapOperator::Reinhard => "Reinhard",
                TonemapOperator::Aces => "ACES",
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut app.tonemapper, TonemapOperator::Aces, "ACES");
                ui.selectable_value(&mut app.tonemapper, TonemapOperator::Reinhard, "Reinhard");
            });
    });

    ui.add_space(10.0);
    ui.separator();
    ui.add_space(10.0);
}
