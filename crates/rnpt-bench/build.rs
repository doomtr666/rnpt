use std::{env, fs, path::PathBuf, process::Command};

const EMBREE_VERSION: &str = "4.4.1";
const EMBREE_ARCHIVE: &str = "embree-4.4.1.x64.windows.zip";
const EMBREE_URL: &str = "https://github.com/RenderKit/embree/releases/download/v4.4.1/embree-4.4.1.x64.windows.zip";

fn main() {
    if env::var("CARGO_FEATURE_EMBREE").is_err() {
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace_dir = manifest_dir.parent().unwrap().parent().unwrap();
    let cache_dir = workspace_dir.join("deps");
    let embree_dir = cache_dir.join(format!("embree-{}", EMBREE_VERSION));

    if !embree_dir.exists() {
        fs::create_dir_all(&cache_dir).expect("failed to create deps/");

        let zip_path = cache_dir.join(EMBREE_ARCHIVE);
        let _ = fs::remove_file(&zip_path);

        eprintln!("[rnpt-bench] Downloading Embree {} ...", EMBREE_VERSION);
        let st = Command::new("curl")
            .args(["-L", "--progress-bar", "--fail", "-o", zip_path.to_str().unwrap(), EMBREE_URL])
            .status()
            .expect("curl not found");
        assert!(st.success(), "curl failed");

        eprintln!("[rnpt-bench] Extracting ...");
        let st = Command::new("powershell")
            .args(["-NoProfile", "-Command",
                &format!("Expand-Archive -Force -Path '{}' -DestinationPath '{}'",
                    zip_path.display(), cache_dir.display())])
            .status()
            .expect("powershell not found");
        assert!(st.success(), "Expand-Archive failed");

        let extracted = fs::read_dir(&cache_dir).unwrap()
            .filter_map(|e| e.ok())
            .find(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false)
                && e.file_name().to_str().unwrap_or("").starts_with("embree-"))
            .unwrap_or_else(|| panic!("no embree-* dir found in deps/"))
            .path();

        fs::rename(&extracted, &embree_dir).unwrap();
        let _ = fs::remove_file(&zip_path);
        eprintln!("[rnpt-bench] Embree ready at {}", embree_dir.display());
    }

    let lib_dir = embree_dir.join("lib");
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=embree4");

    // Copy DLLs next to the bench binary (OUT_DIR/../../.. = target/{profile}/)
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    if let Some(profile_dir) = out_dir.ancestors().nth(3) {
        let bin_dir = embree_dir.join("bin");
        for entry in fs::read_dir(&bin_dir).into_iter().flatten().flatten() {
            let src = entry.path();
            if src.extension().map_or(false, |e| e == "dll") {
                for dir in [profile_dir, &profile_dir.join("deps")] {
                    let dst = dir.join(src.file_name().unwrap());
                    if !dst.exists() { fs::copy(&src, &dst).ok(); }
                }
            }
        }
    }

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", embree_dir.display());
}
