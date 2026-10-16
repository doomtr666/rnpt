//! rnpt-compare: render a scene with several strategies at the same ray budget,
//! compare them against a converged reference, and write an HTML summary.
//!
//!   rnpt-compare --settings renders/<render>.json [options]
//!
//! For each mode (full image, indirect light only): a MIS reference at
//! --ref-rays, every strategy at --rays, then imgcmp (RelMSE, bias, efficiency,
//! error and bias maps). Everything goes to one directory with an index.html.
//! The default directory is keyed by the scene settings, so references are
//! reused across runs as long as the scene, camera, sky and size do not change.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use rnpt_gui::render_io::{self, RenderSettings, RenderStats};

const USAGE: &str = "\
usage: rnpt-compare --settings <render>.json [options]

  --strategies a,b    strategies to compare (default mis,nirc); the first is the
                      efficiency baseline
  --rays N            budget per render, rays per pixel (default 1000)
  --ref-rays N        budget of the MIS reference (default 20000)
  --size WxH          resolution (default: the one in the settings)
  --modes a,b         full, indirect (default both)
  --bias-range R      bias map scale, e.g. 0.2 for +-20% (default 0.2)
  --out DIR           output directory (default renders/compare/<scene>_<WxH>_<settings hash>)
  --imgcmp PATH       imgcmp executable (default tools/imgcmp/target/release/imgcmp)";

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3))
}

fn exe(dir_of: &Path, name: &str) -> PathBuf {
    dir_of.with_file_name(format!("{name}{}", std::env::consts::EXE_SUFFIX))
}

/// Render one image with rnpt-render; returns the path of its .hdr.
fn render(
    rnpt_render: &Path,
    settings_file: &Path,
    strategy: &str,
    rays: u64,
    indirect: bool,
    out: &Path,
) -> Result<PathBuf, String> {
    let mut cmd = Command::new(rnpt_render);
    cmd.arg("--settings").arg(settings_file);
    cmd.args(["--strategy", strategy, "--rays", &rays.to_string(), "--spp", "0"]);
    if indirect {
        cmd.arg("--indirect");
    }
    cmd.arg("--out").arg(out).stderr(Stdio::inherit());
    let output = cmd.output().map_err(|e| format!("{}: {e}", rnpt_render.display()))?;
    if !output.status.success() {
        return Err(format!("rnpt-render failed for {strategy}"));
    }
    // rnpt-render prints the PNG path last; the HDR sits next to it.
    let stdout = String::from_utf8_lossy(&output.stdout);
    let png = stdout.lines().last().ok_or("rnpt-render printed nothing")?.trim().to_string();
    Ok(PathBuf::from(png.strip_suffix(".png").unwrap_or(&png).to_string() + ".hdr"))
}

fn first_hdr(dir: &Path) -> Option<PathBuf> {
    std::fs::read_dir(dir).ok()?.filter_map(|e| e.ok()).map(|e| e.path()).find(|p| p.extension().is_some_and(|x| x == "hdr"))
}

fn rel(path: &Path, base: &Path) -> String {
    path.strip_prefix(base).unwrap_or(path).to_string_lossy().replace('\\', "/")
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let value = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).map(String::as_str);
    let number = |name: &str, default: u64| -> Result<u64, String> {
        value(name).map_or(Ok(default), |v| v.parse().map_err(|_| format!("bad {name}: {v}")))
    };

    let mut settings: RenderSettings = render_io::load_settings(Path::new(value("--settings").ok_or(USAGE)?))?;
    if let Some(size) = value("--size") {
        let (w, h) = size.split_once('x').ok_or("--size is WxH")?;
        settings.width = w.parse().map_err(|_| "bad --size")?;
        settings.height = h.parse().map_err(|_| "bad --size")?;
    }
    let strategies: Vec<String> = value("--strategies").unwrap_or("mis,nirc").split(',').map(|s| {
        render_io::parse_strategy(s).map(|st| render_io::strategy_name(st).to_string())
    }).collect::<Result<_, _>>()?;
    let rays = number("--rays", 1000)?;
    let ref_rays = number("--ref-rays", 20000)?;
    let modes: Vec<String> = value("--modes").unwrap_or("full,indirect").split(',').map(str::to_string).collect();
    let bias_range = value("--bias-range").unwrap_or("0.2");

    // Key the output directory on what defines the image, not on strategy or budget.
    let mut key = settings.clone();
    key.strategy.clear();
    key.rays_per_pixel = 0;
    key.samples_per_pixel = 0;
    key.indirect_only = false;
    let hash = fnv1a(serde_json::to_string(&key).unwrap().as_bytes());
    let scene = Path::new(&settings.scene).file_stem().map_or("scene".into(), |s| s.to_string_lossy().into_owned());
    let out = PathBuf::from(value("--out").map(str::to_string).unwrap_or_else(|| {
        format!("renders/compare/{scene}_{}x{}_{:08x}", settings.width, settings.height, hash as u32)
    }));
    std::fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;

    let settings_file = out.join("settings.json");
    let saved = serde_json::json!({ "settings": key_with(&settings), "stats": RenderStats::default() });
    std::fs::write(&settings_file, serde_json::to_string_pretty(&saved).unwrap()).map_err(|e| e.to_string())?;

    let self_exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let rnpt_render = exe(&self_exe, "rnpt-render");
    let imgcmp = value("--imgcmp").map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(format!("tools/imgcmp/target/release/imgcmp{}", std::env::consts::EXE_SUFFIX))
    });
    if !imgcmp.exists() {
        return Err(format!("{} not found: build tools/imgcmp first, or pass --imgcmp", imgcmp.display()));
    }

    let mut sections = String::new();
    for mode in &modes {
        let indirect = match mode.as_str() {
            "full" => false,
            "indirect" => true,
            m => return Err(format!("unknown mode {m} (full, indirect)")),
        };
        let dir = out.join(mode);
        eprintln!("\n== {mode} ==");

        let ref_dir = dir.join(format!("ref-mis-{ref_rays}"));
        let reference = match first_hdr(&ref_dir) {
            Some(r) => {
                eprintln!("reference: reusing {}", r.display());
                r
            }
            None => render(&rnpt_render, &settings_file, "mis", ref_rays, indirect, &ref_dir)?,
        };

        let run_dir = dir.join(format!("{rays}rpp"));
        let _ = std::fs::remove_dir_all(&run_dir);
        let mut renders = vec![];
        for s in &strategies {
            renders.push((s.clone(), render(&rnpt_render, &settings_file, s, rays, indirect, &run_dir)?));
        }

        let metrics = run_dir.join("metrics.json");
        let mut cmd = Command::new(&imgcmp);
        cmd.arg(&reference);
        for (_, r) in &renders {
            cmd.arg(r);
        }
        cmd.arg("--out").arg(run_dir.join("maps")).arg("--json").arg(&metrics).args(["--bias-range", bias_range]);
        if !cmd.status().map_err(|e| format!("{}: {e}", imgcmp.display()))?.success() {
            return Err("imgcmp failed".into());
        }
        sections.push_str(&html_section(mode, &out, &reference, &renders, &metrics)?);
    }

    let index = out.join("index.html");
    std::fs::write(&index, html_page(&settings, rays, ref_rays, &sections)).map_err(|e| e.to_string())?;
    println!("{}", index.display());
    Ok(())
}

/// The settings as rendered: strategy and budget come from the command line.
fn key_with(settings: &RenderSettings) -> RenderSettings {
    let mut s = settings.clone();
    s.rays_per_pixel = 0;
    s.samples_per_pixel = 0;
    s.indirect_only = false;
    s
}

fn html_section(
    mode: &str,
    out: &Path,
    reference: &Path,
    renders: &[(String, PathBuf)],
    metrics: &Path,
) -> Result<String, String> {
    let doc: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(metrics).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let rows = doc["renders"].as_array().ok_or("bad metrics.json")?;
    let png = |hdr: &Path| hdr.with_extension("png");
    let num = |v: &serde_json::Value, f: &dyn Fn(f64) -> String| v.as_f64().map_or("-".into(), f);

    let mut table = String::from(
        "<table><tr><th>strategy</th><th>rays/px</th><th>spp</th><th>time</th><th>RelMSE</th><th>bias</th>\
         <th>efficiency / ray</th><th>efficiency / s</th></tr>",
    );
    let mut grid = format!(
        "<div class=\"row\"><figure><img src=\"{}\"><figcaption>reference (MIS)</figcaption></figure></div>",
        rel(&png(reference), out)
    );
    for ((name, hdr), row) in renders.iter().zip(rows) {
        table.push_str(&format!(
            "<tr><td>{name}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
            num(&row["rays_per_pixel"], &|v| format!("{v:.0}")),
            num(&row["samples_per_pixel"], &|v| format!("{v:.1}")),
            num(&row["seconds"], &|v| format!("{v:.2} s")),
            num(&row["rel_mse"], &|v| format!("{v:.3e}")),
            num(&row["bias"], &|v| format!("{:+.2}%", v * 100.0)),
            num(&row["efficiency_per_ray"], &|v| format!("{v:.2}x")),
            num(&row["efficiency_per_second"], &|v| format!("{v:.2}x")),
        ));
        let map = |key: &str| row[key].as_str().map(|p| rel(Path::new(p), out)).unwrap_or_default();
        grid.push_str(&format!(
            "<div class=\"row\"><figure><img src=\"{}\"><figcaption>{name}</figcaption></figure>\
             <figure><img src=\"{}\"><figcaption>{name}: relative error</figcaption></figure>\
             <figure><img src=\"{}\"><figcaption>{name}: bias</figcaption></figure></div>",
            rel(&png(hdr), out),
            map("relerr_map"),
            map("bias_map"),
        ));
    }
    table.push_str("</table>");
    let range = doc["bias_range"].as_f64().unwrap_or(0.2) * 100.0;
    Ok(format!(
        "<h2>{mode}</h2>{table}<p class=\"legend\">Efficiency is 1 / (RelMSE &times; cost), relative to the first \
         strategy. Relative error: log scale, 1e-4 (black) to 1 (white). Bias: blue too dark, red too bright, \
         &plusmn;{range:.0}%, Gaussian-averaged so noise cancels out.</p>{grid}"
    ))
}

fn html_page(settings: &RenderSettings, rays: u64, ref_rays: u64, sections: &str) -> String {
    let env = settings.env.as_ref().map_or("none".into(), |e| format!("{} (x{})", e.path, e.intensity));
    format!(
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>rnpt comparison</title><style>\
         body{{font-family:sans-serif;background:#1b1b1b;color:#ddd;margin:24px}}\
         table{{border-collapse:collapse;margin:12px 0}}td,th{{border:1px solid #444;padding:4px 10px;text-align:right}}\
         td:first-child,th:first-child{{text-align:left}}.row{{display:flex;gap:8px;flex-wrap:wrap}}\
         figure{{margin:0 0 8px 0}}img{{width:360px;image-rendering:pixelated}}figcaption{{font-size:13px;color:#aaa}}\
         .legend{{color:#999;font-size:13px}}</style></head><body>\
         <h1>{scene}</h1><p>{w}x{h}, sky: {env}, exposure {exp}. Each strategy gets {rays} rays per pixel; \
         the reference is MIS at {ref_rays} rays per pixel.</p>{sections}</body></html>",
        scene = settings.scene,
        w = settings.width,
        h = settings.height,
        exp = settings.exposure,
    )
}
