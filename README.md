# rnpt — Rust Path Tracer

A physically-based CPU path tracer written in Rust. The primary goal is to prototype and validate advanced rendering algorithms in a clean, readable codebase before porting them to GPU. It doubles as an educational reference: every major system (BVH, BRDF, MIS, NIRC) is implemented from scratch with correctness and clarity as first-class concerns.

## Goals

- **Algorithm prototyping** — validate rendering research ideas (ReSTIR, neural caches, MIS variants) on CPU where iteration is fast and debugging is straightforward, then port the validated design to GPU (CUDA/WGSL).
- **Educational reference** — each subsystem is self-contained and documented. The BVH in particular is production-quality and serves as a study case for SIMD-accelerated tree traversal.
- **Real-world scenes** — glTF 2.0 import with PBR materials, punctual lights, HDRI environments, transmission/IOR/volume extensions. Tested on Cornell Box, Bistro exterior, and others.

## Features

### Rendering

| Feature | Details |
|---|---|
| Integrator | Unbiased path tracing, Russian roulette termination (PBRT-style, unbiased) |
| Sampling strategies | BRDF-only · NEE-only · MIS (balance heuristic) · Direct-only · NIRC |
| BRDF | Lambertian diffuse + GGX Cook-Torrance specular (metallic-roughness) |
| Glass | Thin/thick dielectric, Schlick Fresnel, GGX microfacet sampling |
| Area lights | Emissive mesh triangles, alias-table O(1) selection |
| Punctual lights | Point and directional (`KHR_lights_punctual`) |
| Environment | HDRI equirectangular, 2D piecewise-constant importance sampling |

### BVH

8-wide BVH (BVH-8) built with SAH (16-bin sweep). Nodes store 8 child AABBs in SoA layout with `#[repr(C, align(32))]` for AVX2-friendly loads. Triangle intersection uses Möller–Trumbore with `wide::f32x8` fused multiply-add — 8 triangles tested in parallel per SIMD lane. Octant-ordered child visitation for front-to-back traversal.

The BVH lives in its own crate (`rnpt-bvh`) and can be used independently.

### NIRC — Neural Incident Radiance Cache

An online-trained neural cache that replaces indirect path tracing bounces with a network prediction, trading a small bias for significantly higher convergence speed.

- **Architecture**: 4-layer MLP, hidden dim 16, SiLU activations. ~755 parameters, <3 KB — fits entirely in L1 cache.
- **Encoding**: NeRF sinusoidal + one-blob positional encoding for position and direction (`INPUT_DIM = 9`).
- **Training**: Adam optimizer, log-space MSE loss for HDR balance, EMA-smoothed inference snapshot published to workers lock-free.
- **Sample collection**: Lock-free ring buffer (131 072 entries). Every 64th rendered pixel triggers a dedicated MIS path; all bounce-level radiances are pushed as training targets.
- **Inference**: `gemv`-based forward pass (no weight matrix copies), alternating activation buffers. Backward pass uses `gemv_tr` to transpose-multiply without materializing the transpose.
- **Quality**: ~3.3% RelMean on Cornell Box, ~3.4% on Bistro exterior (vs MIS ground truth), at 2× the path throughput of pure MIS.

### Parallel rendering

Workers run on all available cores, each owning a disjoint set of 128×128 pixel tiles. The pixel buffer is shared lock-free (workers write disjoint tiles; the GUI reads concurrently for progressive display). NIRC network hot-swap happens without restarting a render pass.

## Crate structure

```
rnpt/
├── crates/
│   ├── rnpt/          # Core library — path tracing, materials, NIRC, scene model
│   ├── rnpt-bvh/      # Standalone BVH — SAH builder, 8-wide SIMD traversal
│   └── rnpt-gui/      # eframe/egui GUI — scene import, live render, NIRC controls
└── assets/            # glTF scenes, HDRI environment maps
```

## Scene format

glTF 2.0 (`.glb` / `.gltf`) via the `gltf` crate. Supported extensions:

- `KHR_lights_punctual`
- `KHR_materials_emissive_strength`
- `KHR_materials_transmission`
- `KHR_materials_ior`
- `KHR_materials_volume`

HDRI environment maps (`.hdr`) loaded via the `image` crate.

## Building

```bash
cargo build --release
cargo run --release -p rnpt-gui
```

Requires a recent stable Rust toolchain. AVX2 is used automatically where available via the `wide` crate.

## GUI controls

| Input | Action |
|---|---|
| Left drag | Orbit camera |
| Right drag | Pan |
| Scroll | Zoom |
| Ctrl+click | Place NIRC directional probe at surface point |
| R | Reset render |
