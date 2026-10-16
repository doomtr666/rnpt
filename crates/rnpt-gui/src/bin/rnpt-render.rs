//! Headless renderer: same tracer and outputs as the GUI, without a window.
//!
//!   rnpt-render --settings renders/<render>.json [overrides]
//!   rnpt-render --scene assets/cornell.glb [--env assets/sky.hdr] [overrides]
//!
//! Overrides: --strategy S, --indirect, --rays N, --spp N, --seconds T, --size WxH, --out DIR.
//! Stops when every pixel spent its budget (--rays / --spp), or after --seconds.
//! Writes <out>/<name>.png, .hdr and .json like the GUI's "Save image", with
//! timing details in the .json to tell rendering, training and waiting apart.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use rnpt_gui::render_io::{self, CameraSettings, EnvSettings, RenderSettings, RenderStats};

const USAGE: &str = "\
usage: rnpt-render (--settings <render>.json | --scene <file.glb>) [options]

  --settings F        re-render with the settings saved next to an image
  --scene F           scene to render (first camera of the scene, or default)
  --env F             environment map (.hdr); --env-intensity X, --env-rotation DEG
  --strategy S        nirc, mis, nee, brdf, direct
  --indirect          keep only indirect light (2+ diffuse/glossy bounces)
  --rays N            budget in rays per pixel
  --spp N             budget in samples per pixel
  --seconds T         stop after T seconds (equal-time comparisons)
  --size WxH          resolution (default 800x800)
  --out DIR           output directory (default renders)";

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn settings_from_scene(scene: &str) -> Result<RenderSettings, String> {
    // Same camera as the GUI on load: the scene's first camera, else the default.
    let (loaded, _) = rnpt_gui::asset_importer::import_scene(scene).map_err(|e| format!("{scene}: {e}"))?;
    let cam = loaded.cameras.first().cloned().unwrap_or_default();
    Ok(RenderSettings {
        scene: scene.into(),
        camera: CameraSettings {
            position: [cam.position.x, cam.position.y, cam.position.z],
            target: [cam.target.x, cam.target.y, cam.target.z],
            fov: cam.fov,
        },
        width: 800,
        height: 800,
        strategy: "mis".into(),
        rays_per_pixel: 0,
        samples_per_pixel: 0,
        env: None,
        exposure: 1.0,
        tonemapper: "aces".into(),
        indirect_only: false,
    })
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let value = |name: &str| -> Option<&str> {
        args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).map(String::as_str)
    };
    let parsed = |name: &str| -> Result<Option<f64>, String> {
        value(name).map(|v| v.parse::<f64>().map_err(|_| format!("bad {name}: {v}"))).transpose()
    };

    let mut settings = match (value("--settings"), value("--scene")) {
        (Some(f), _) => render_io::load_settings(Path::new(f))?,
        (None, Some(scene)) => settings_from_scene(scene)?,
        (None, None) => return Err(USAGE.into()),
    };
    if let Some(f) = value("--env") {
        settings.env = Some(EnvSettings {
            path: f.into(),
            intensity: parsed("--env-intensity")?.unwrap_or(1.0) as f32,
            rotation_deg: parsed("--env-rotation")?.unwrap_or(0.0) as f32,
        });
    }
    if let Some(s) = value("--strategy") {
        settings.strategy = render_io::strategy_name(render_io::parse_strategy(s)?).into();
    }
    if let Some(n) = parsed("--rays")? {
        settings.rays_per_pixel = n as u64;
    }
    if let Some(n) = parsed("--spp")? {
        settings.samples_per_pixel = n as u32;
    }
    if let Some(size) = value("--size") {
        let (w, h) = size.split_once('x').ok_or("--size is WxH")?;
        settings.width = w.parse().map_err(|_| "bad --size")?;
        settings.height = h.parse().map_err(|_| "bad --size")?;
    }
    if args.iter().any(|a| a == "--indirect") {
        settings.indirect_only = true;
    }
    let max_seconds = parsed("--seconds")?;
    let out = PathBuf::from(value("--out").unwrap_or("renders"));
    if settings.rays_per_pixel == 0 && settings.samples_per_pixel == 0 && max_seconds.is_none() {
        return Err("no stop condition: give --rays, --spp or --seconds".into());
    }

    eprintln!(
        "{} | {}{} | {}x{} | budget {} rays/px, {} spp{}",
        settings.scene,
        settings.strategy,
        if settings.indirect_only { " (indirect only)" } else { "" },
        settings.width,
        settings.height,
        settings.rays_per_pixel,
        settings.samples_per_pixel,
        max_seconds.map_or(String::new(), |s| format!(", {s} s max"))
    );
    let config = render_io::load_config(&settings)?;
    let budget = config.budget;

    let tracer = rnpt::ParallelTracer::new(config);
    let mut pixels = vec![rnpt::Pixel::default(); settings.width * settings.height];
    let (mut paths, mut rays) = (0u64, 0u64);
    let num_pixels = pixels.len() as u64;
    let start = Instant::now();
    let mut last_report = start;
    // Poll every millisecond so the measured time ends when the render does.
    let seconds = loop {
        // Drive NIRC training like the GUI's frame loop does (no-op otherwise).
        tracer.train_nirc();
        std::thread::sleep(Duration::from_millis(1));

        let now = Instant::now();
        let elapsed = (now - start).as_secs_f64();
        let done = !budget.is_unlimited() && tracer.pixels_done() >= num_pixels;
        if done || max_seconds.is_some_and(|s| elapsed >= s) {
            break elapsed;
        }
        if now - last_report >= Duration::from_millis(250) {
            last_report = now;
            tracer.fetch_pixels(&mut pixels);
            paths += tracer.pop_rays_traced();
            rays += tracer.pop_real_rays_traced() + tracer.pop_shadow_rays_traced();
            let (rpp, spp) = render_io::pixel_averages(&pixels);
            eprint!("\r  {elapsed:6.1} s  {rpp:8.1} rays/px  {spp:7.1} spp   ");
        }
    };
    tracer.fetch_pixels(&mut pixels);
    paths += tracer.pop_rays_traced();
    rays += tracer.pop_real_rays_traced() + tracer.pop_shadow_rays_traced();
    let timing = tracer.timing();
    drop(tracer);
    eprintln!();

    let (rays_per_pixel, samples_per_pixel) = render_io::pixel_averages(&pixels);
    let stats = RenderStats {
        rays_per_pixel,
        samples_per_pixel,
        seconds,
        paths: Some(paths),
        rays: Some(rays),
        threads: Some(timing.threads),
        nirc_steps: Some(timing.nirc_steps),
        train_s: Some(timing.train_s),
        pacing_wait_s: Some(timing.pacing_wait_s),
    };
    let png = render_io::save_render(&out, &settings, &stats, &pixels)?;

    let wall = seconds * timing.threads as f64;
    eprintln!("  {:.2} Mpaths/s, {:.2} Mrays/s", paths as f64 / seconds / 1e6, rays as f64 / seconds / 1e6);
    if timing.nirc_steps > 0 {
        eprintln!(
            "  NIRC: {} training steps, training {:.2} s ({:.0}% of the run), workers waiting {:.1}% of their time",
            timing.nirc_steps,
            timing.train_s,
            100.0 * timing.train_s / seconds,
            100.0 * timing.pacing_wait_s / wall
        );
    }
    println!("{}", png.display());
    Ok(())
}
