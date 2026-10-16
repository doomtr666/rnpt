//! imgcmp: compare renders against a reference, at equal cost.
//!
//!   imgcmp <reference.hdr> <render.hdr>... [--out DIR] [--json FILE] [--bias-range R] [--bias-sigma S]
//!
//! For each render: RelMSE, global bias, efficiency (1 / (RelMSE x cost)), and
//! two maps written as PNG: the relative error (log scale) and the local bias
//! (signed error averaged over a window, so that noise cancels out and only
//! systematic error remains).

use std::path::{Path, PathBuf};

/// Usual RelMSE regularization: error^2 / (ref^2 + EPS).
const REL_EPS: f32 = 0.01;
/// Log range of the relative-error map: 1e-4 (black) .. 1 (white).
const ERR_LOG_MIN: f32 = -4.0;
const ERR_LOG_MAX: f32 = 0.0;

struct Image {
    w: u32,
    h: u32,
    rgb: Vec<[f32; 3]>,
}

fn load(path: &Path) -> Result<Image, String> {
    let img = image::open(path).map_err(|e| format!("{}: {e}", path.display()))?.into_rgb32f();
    let (w, h) = img.dimensions();
    let rgb = img.pixels().map(|p| p.0).collect();
    Ok(Image { w, h, rgb })
}

fn luminance(c: [f32; 3]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

/// Cost recorded in rnpt's file names: `<scene>_<strategy>_<rays>rpp_<spp>spp_<time>s_<stamp>`.
#[derive(Default)]
struct Cost {
    rpp: Option<f64>,
    spp: Option<f64>,
    seconds: Option<f64>,
}

fn parse_cost(path: &Path) -> Cost {
    let stem = path.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let stem = stem.strip_suffix(".hdr").unwrap_or(&stem);
    let mut cost = Cost::default();
    for tok in stem.split('_') {
        let num = |suffix: &str| tok.strip_suffix(suffix).and_then(|n| n.parse::<f64>().ok());
        if let Some(v) = num("rpp") {
            cost.rpp = Some(v);
        } else if let Some(v) = num("spp") {
            cost.spp = Some(v);
        } else if let Some(v) = num("s") {
            cost.seconds = Some(v);
        }
    }
    cost
}

struct Metrics {
    rel_mse: f64,
    /// Mean over pixels and channels of (render - ref) / (ref + REL_EPS): every
    /// pixel weighs the same, so the lamp and the sky do not drown the rest.
    /// Zero on average for an unbiased estimator.
    bias: f64,
}

fn metrics(img: &Image, reference: &Image) -> Metrics {
    let (mut rel, mut bias) = (0.0f64, 0.0f64);
    for (x, r) in img.rgb.iter().zip(&reference.rgb) {
        for c in 0..3 {
            let d = x[c] - r[c];
            rel += (d * d / (r[c] * r[c] + REL_EPS)) as f64;
            bias += (d / (r[c] + REL_EPS)) as f64;
        }
    }
    let n = (img.rgb.len() * 3) as f64;
    Metrics { rel_mse: rel / n, bias: bias / n }
}

/// Magma-like ramp, t in [0, 1].
fn magma(t: f32) -> [u8; 3] {
    const STOPS: [[f32; 3]; 5] = [
        [0.0, 0.0, 4.0],
        [80.0, 18.0, 123.0],
        [183.0, 55.0, 121.0],
        [252.0, 137.0, 97.0],
        [252.0, 253.0, 191.0],
    ];
    let x = t.clamp(0.0, 1.0) * 4.0;
    let i = (x as usize).min(3);
    let f = x - i as f32;
    std::array::from_fn(|c| (STOPS[i][c] + (STOPS[i + 1][c] - STOPS[i][c]) * f) as u8)
}

/// Blue (negative) - white (0) - red (positive), v in [-1, 1].
fn diverging(v: f32) -> [u8; 3] {
    let v = v.clamp(-1.0, 1.0);
    let fade = |a: f32| (255.0 * (1.0 - a.abs())) as u8;
    if v < 0.0 {
        [fade(v), fade(v), 255]
    } else {
        [255, fade(v), fade(v)]
    }
}

fn save_png(path: &Path, w: u32, h: u32, rgb: &[[u8; 3]]) -> Result<(), String> {
    let flat: Vec<u8> = rgb.iter().flatten().copied().collect();
    image::save_buffer(path, &flat, w, h, image::ExtendedColorType::Rgb8).map_err(|e| format!("{}: {e}", path.display()))
}

fn error_map(img: &Image, reference: &Image) -> Vec<[u8; 3]> {
    img.rgb
        .iter()
        .zip(&reference.rgb)
        .map(|(x, r)| {
            let e = (0..3).map(|c| (x[c] - r[c]).powi(2) / (r[c] * r[c] + REL_EPS)).sum::<f32>() / 3.0;
            let t = (e.max(1e-12).log10() - ERR_LOG_MIN) / (ERR_LOG_MAX - ERR_LOG_MIN);
            magma(t)
        })
        .collect()
}

/// Separable Gaussian blur of a w x h scalar field (edges clamped).
fn gaussian_blur(field: &[f64], w: usize, h: usize, sigma: f64) -> Vec<f64> {
    let r = (3.0 * sigma).ceil() as isize;
    let kernel: Vec<f64> = (-r..=r).map(|i| (-(i * i) as f64 / (2.0 * sigma * sigma)).exp()).collect();
    let norm: f64 = kernel.iter().sum();
    let pass = |src: &[f64], horizontal: bool| -> Vec<f64> {
        let mut dst = vec![0.0; w * h];
        for y in 0..h {
            for x in 0..w {
                let mut acc = 0.0;
                for (k, wk) in kernel.iter().enumerate() {
                    let o = k as isize - r;
                    let (sx, sy) = if horizontal {
                        ((x as isize + o).clamp(0, w as isize - 1) as usize, y)
                    } else {
                        (x, (y as isize + o).clamp(0, h as isize - 1) as usize)
                    };
                    acc += wk * src[sy * w + sx];
                }
                dst[y * w + x] = acc / norm;
            }
        }
        dst
    };
    pass(&pass(field, true), false)
}

/// Signed relative luminance error, averaged with a Gaussian of `sigma` pixels:
/// noise averages out, systematic bias stays. Gaussian rather than a box, so
/// the map has no square artifacts (a box turns every outlier into a square).
fn bias_map(img: &Image, reference: &Image, sigma: f32, range: f32) -> Vec<[u8; 3]> {
    let (w, h) = (img.w as usize, img.h as usize);
    let diff: Vec<f64> = img.rgb.iter().zip(&reference.rgb).map(|(x, r)| (luminance(*x) - luminance(*r)) as f64).collect();
    let refl: Vec<f64> = reference.rgb.iter().map(|r| luminance(*r) as f64).collect();
    let (diff, refl) = (gaussian_blur(&diff, w, h, sigma as f64), gaussian_blur(&refl, w, h, sigma as f64));
    diff.iter().zip(&refl).map(|(d, r)| diverging((d / (r + REL_EPS as f64)) as f32 / range)).collect()
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut paths: Vec<PathBuf> = vec![];
    let mut out_dir: Option<PathBuf> = None;
    let mut json_path: Option<PathBuf> = None;
    let mut bias_range = 0.1f32;
    let mut bias_sigma = 6.0f32;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--out" => out_dir = Some(args.next().ok_or("--out needs a directory")?.into()),
            "--json" => json_path = Some(args.next().ok_or("--json needs a file")?.into()),
            "--bias-range" => bias_range = args.next().and_then(|v| v.parse().ok()).ok_or("bad --bias-range")?,
            "--bias-sigma" => bias_sigma = args.next().and_then(|v| v.parse().ok()).ok_or("bad --bias-sigma")?,
            _ => paths.push(a.into()),
        }
    }
    if paths.len() < 2 {
        return Err("usage: imgcmp <reference.hdr> <render.hdr>... [--out DIR] [--json FILE] [--bias-range R] [--bias-sigma S]".into());
    }

    let reference = load(&paths[0])?;
    println!("reference: {} ({}x{})", paths[0].display(), reference.w, reference.h);
    println!(
        "\n{:<44} {:>8} {:>7} {:>8} {:>10} {:>8} {:>10} {:>10}",
        "render", "rays/px", "spp", "time s", "RelMSE", "bias", "eff/ray", "eff/s"
    );

    let mut base_eff: Option<(Option<f64>, Option<f64>)> = None;
    let mut rows = vec![];
    for path in &paths[1..] {
        let img = load(path)?;
        if (img.w, img.h) != (reference.w, reference.h) {
            return Err(format!("{}: {}x{}, reference is {}x{}", path.display(), img.w, img.h, reference.w, reference.h));
        }
        let m = metrics(&img, &reference);
        let cost = parse_cost(path);

        // Efficiency = 1 / (error x cost), shown relative to the first render.
        // An image identical to the reference has no finite efficiency.
        let finite = m.rel_mse > 0.0;
        let eff_ray = cost.rpp.filter(|_| finite).map(|c| 1.0 / (m.rel_mse * c));
        let eff_s = cost.seconds.filter(|_| finite).map(|c| 1.0 / (m.rel_mse * c));
        let (b_ray, b_s) = if finite { *base_eff.get_or_insert((eff_ray, eff_s)) } else { (None, None) };
        let rel = |e: Option<f64>, b: Option<f64>| match (e, b) {
            (Some(e), Some(b)) => format!("{:.2}x", e / b),
            _ => "-".into(),
        };
        let opt = |v: Option<f64>, prec: usize| v.map_or("-".into(), |v| format!("{v:.prec$}"));

        let name = path.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        println!(
            "{:<44} {:>8} {:>7} {:>8} {:>10.3e} {:>+7.2}% {:>10} {:>10}",
            name,
            opt(cost.rpp, 0),
            opt(cost.spp, 1),
            opt(cost.seconds, 2),
            m.rel_mse,
            m.bias * 100.0,
            rel(eff_ray, b_ray),
            rel(eff_s, b_s)
        );

        let dir = out_dir.clone().unwrap_or_else(|| path.parent().map(Path::to_path_buf).unwrap_or_default());
        std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let stem = name.strip_suffix(".hdr").unwrap_or(&name);
        let relerr_png = dir.join(format!("{stem}.relerr.png"));
        let bias_png = dir.join(format!("{stem}.bias.png"));
        save_png(&relerr_png, img.w, img.h, &error_map(&img, &reference))?;
        save_png(&bias_png, img.w, img.h, &bias_map(&img, &reference, bias_sigma, bias_range))?;

        let ratio = |e: Option<f64>, b: Option<f64>| e.zip(b).map(|(e, b)| e / b);
        rows.push(serde_json::json!({
            "render": path,
            "rays_per_pixel": cost.rpp,
            "samples_per_pixel": cost.spp,
            "seconds": cost.seconds,
            "rel_mse": m.rel_mse,
            "bias": m.bias,
            "efficiency_per_ray": ratio(eff_ray, b_ray),
            "efficiency_per_second": ratio(eff_s, b_s),
            "relerr_map": relerr_png,
            "bias_map": bias_png,
        }));
    }

    if let Some(path) = json_path {
        let doc = serde_json::json!({
            "reference": paths[0],
            "bias_sigma_px": bias_sigma,
            "bias_range": bias_range,
            "renders": rows,
        });
        std::fs::write(&path, serde_json::to_string_pretty(&doc).unwrap()).map_err(|e| format!("{}: {e}", path.display()))?;
    }

    println!(
        "\nmaps: <render>.relerr.png = per-pixel RelMSE, log scale 1e-4 (black) .. 1 (white)\n      \
         <render>.bias.png = local relative bias (Gaussian, sigma {} px), blue -{:.0}% .. white 0 .. red +{:.0}%",
        bias_sigma,
        bias_range * 100.0,
        bias_range * 100.0
    );
    Ok(())
}
