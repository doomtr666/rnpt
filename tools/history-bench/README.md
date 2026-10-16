# history-bench

Replays the git history of rnpt's BVH and measures every version on identical inputs, so the
optimization work can be looked at step by step: what each commit changed, how much it gained, and
whether it was still correct.

```sh
cd tools/history-bench
cargo run --release -- run        # prepare inputs, measure what is missing, write the report
cargo run --release -- report     # rebuild the report from cached results only
cargo run --release -- list       # commits that would be measured
cargo run --release -- run --branch bvh-history   # another branch's history, results apart
```

Results land in `results/<cpu>/<mode>/`: one JSON file per commit, plus `REPORT.md` and
`results.csv` generated from them.

## How it works

1. **Inputs.** Each scene is generated or imported once, flattened to a world-space triangle soup,
   and stored in the cache with three ray sets: `primary` (camera rays, coherent), `diffuse`
   (cosine-sampled bounces from the primary hits, incoherent) and `shadow` (from the primary hits
   to an area light above the scene). The first `--check` rays of each set get a brute-force
   reference. Everything is keyed by content hash.
2. **Per commit.** The tree is extracted with `git archive`, and a small adapter crate is written
   next to it. The adapter is built against that commit's own sources and `Cargo.lock`. The
   adapter matching the commit's API is chosen by trying to compile: `adapters/era_a.rs` for the
   BVH inside the `rnpt` crate (with `textures` / `occluded` features for the API changes),
   `adapters/era_b.rs` for the `rnpt-bvh` crate.
3. **Measurement.** Single thread, pinned to one core (`--core`), high priority. Each measurement
   runs in `--procs` separate processes (memory layout varies per process, which dominates the noise
   on scenes larger than the cache); the report gives the median over processes. In each process: one warm-up pass,
   then `--passes` timed passes over every ray of the set; the median is reported. The first
   `--check` results are compared with the brute-force reference, and a version that disagrees on
   more than 0.1 % of them is flagged ✗ in the report: a fast wrong answer does not count.

## Reproducibility

Every result carries a key hashing everything that can change it: the adapter and `common` sources,
the input files, the rustc version, the build mode and its flags, the pass and check counts.
`run` only measures the (commit, scene, ray set) triples whose key is missing or stale, so fixing
an adapter, adding a scene or switching toolchain re-measures exactly what is affected, unattended.
`--force` re-measures everything selected.

Build modes:

- `--mode own` (default): each commit is built as it was, with its own `[profile.release]` and
  `.cargo/config.toml`. The history then includes changes to the build itself, such as the switch
  to `-C target-cpu=native` and LTO in `25fe426`.
- `--mode head`: every commit is built with today's profile and `-C target-cpu=native`, to isolate
  algorithmic changes from build changes.

The cache lives in `%LOCALAPPDATA%/rnpt-history-bench` (or `~/.cache/rnpt-history-bench`), see
`--cache`. Deleting it only costs time. Bump `INPUTS_VERSION` (driver/src/inputs.rs) when a scene
or ray generator changes, and `PROTOCOL_VERSION` (driver/src/runner.rs) when the measurement
changes without the sources changing.

## Scenes

| name | source |
|---|---|
| `hf-100k`, `hf-1m` (`hf-10m` with `--big`) | generated height field |
| `cluster-100k`, `cluster-1m` (`cluster-10m` with `--big`) | generated cluster of overlapping spheres |
| `sponza`, `bistro-ext` | `assets/sponza.glb`, `assets/bistro_ext.glb`, skipped when missing |

Primary hits used to shape the secondary rays are found with today's `rnpt-bvh`. That only affects
the inputs, which are then frozen and shared by every commit; correctness is always checked against
brute force.

## Limits

- Only throughput, build time and correctness are measured. Per-ray counters (nodes visited,
  triangles tested) would need patching every historical version, so they are left to the current
  version.
- Thread pinning is implemented on Windows only; elsewhere the result records `pinned: false`.
