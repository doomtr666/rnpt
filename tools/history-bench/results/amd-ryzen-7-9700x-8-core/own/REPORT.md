# BVH history — amd-ryzen-7-9700x-8-core — build config: own

Single-thread throughput in Mrays/s: median over processes of the median pass. ✗ = results disagree with the brute-force reference; † = no occlusion query yet, shadow rays use closest-hit.

## bistro-ext

| commit | date | subject | primary | diffuse | shadow | ortho | build ms |
|---|---|---|---:|---:|---:|---:|---:|
| `ab81978` | 2026-06-12 | add basic bvh | 1.2 | 0.2 | 0.2 † | 0.6 | 986 |
| `2ebe9f1` | 2026-06-13 | various improvements | 1.0 | 0.2 | 0.2 † | 0.6 | 982 |
| `25fe426` | 2026-06-13 | intersection simd optimization with wide | 1.4 | 0.4 | 0.4 † | 1.0 | 943 |
| `124a279` | 2026-06-13 | change the bvh algorithm | 1.5 | 0.4 | 0.4 † | 1.1 | 726 |
| `3477fb2` | 2026-06-13 | modified SAH, clustering instead of morton for SOA triangles grouping | 5.7 | 1.4 | 1.5 † | 4.7 | 1029 |
| `db51ac1` | 2026-06-13 | cleanup comments, and magic numbers | 5.9 | 1.5 | 1.6 † | 4.7 | 1040 |
| `b1fb0ba` | 2026-06-13 | tweaks | 5.9 | 1.5 | 1.5 † | 4.7 | 1085 |
| `0d2606b` | 2026-06-13 | add textures support | 5.9 | 1.4 | 1.5 † | 4.7 | 1143 |
| `0778479` | 2026-06-13 | fix warnings | 5.9 | 1.4 | 1.5 † | 4.6 | 1158 |
| `ec4fa65` | 2026-06-14 | optimize bvh | 6.5 | 1.6 | 1.7 † | 4.9 | 1135 |
| `73eb3ff` | 2026-06-14 | slight bvh optimisation | 7.3 | 1.8 | 1.9 † | 5.7 | 1126 |
| `c0e2370` | 2026-06-14 | bvh8 attempt | 11.5 | 3.2 | 3.4 † | 8.4 | 1146 |
| `51ab953` | 2026-06-14 | mix performance tweaks | 11.2 | 3.1 | 3.3 † | 7.9 | 1127 |
| `59e8bde` | 2026-06-14 | shadow ray optimization | 11.1 | 3.1 | 4.9 | 7.9 | 1153 |
| `65c2817` | 2026-06-15 | refactorings | 11.0 | 3.1 | 4.9 | 7.9 | 1143 |
| `b99763c` | 2026-06-15 | emissive lights | 11.2 | 3.1 | 4.8 | 7.9 | 1132 |
| `19f08c5` | 2026-06-15 | generic light list (emissive mesh + anlytic) | 11.0 | 3.1 | 4.8 | 7.8 | 1136 |
| `f0d9350` | 2026-06-16 | cleanup, add comments | 11.0 | 3.1 | 4.8 | 7.8 | 1129 |
| `d73438c` | 2026-06-16 | add MIS support | 11.2 | 3.1 | 4.8 | 7.9 | 1156 |
| `3ce2ba2` | 2026-06-17 | add hdr sky support | 11.1 | 3.1 | 4.8 | 7.9 | 1153 |
| `18cc184` | 2026-06-17 | add benchmarks | 11.2 | 3.1 | 4.8 | 7.9 | 1159 |
| `befb868` | 2026-06-18 | try octant order lut to avoid sortir children in bvh traversal | 10.3 | 2.7 | 4.5 | 7.7 | 1147 |
| `4a099c0` | 2026-06-18 | extract BVH into rnpt-bvh crate with clean Embree-like interface | 9.9 | 2.7 | 4.4 | 7.4 | 557 |
| `fb5869f` | 2026-06-19 | Cook Torrance/GGX and PBR materials support. | 9.9 | 2.7 | 4.4 | 7.4 | 558 |
| `e8f107d` | 2026-06-20 | glass support | 9.4 | 2.6 | 4.1 | 7.0 | 561 |
| `d17de5d` | 2026-06-23 | use FMA for ray intersection | 9.8 | 2.7 | 4.4 | 0.0 | 561 |
| `1c9d44d` | 2026-06-24 | true restir, fix regression in bvh intersection | 9.7 | 2.7 | 4.3 | 7.3 | 563 |
| `63042ea` | 2026-06-26 | cleanup | 9.7 | 2.6 | 4.3 | 7.4 | 564 |
| `c020143` | 2026-06-27 | first nirc version | 9.8 | 2.6 | 4.3 | 7.3 | 564 |
| `29cac5b` | 2026-07-16 | cleanup before going public | 9.8 | 2.7 | 4.3 | 7.4 | 564 |
| `75e6c3a` | 2026-07-16 | remove dead code | 9.8 | 2.7 | 4.3 | 7.4 | 564 |
| `cbd3127` | 2026-10-15 | bvh: revert the octant LUT (befb868), sort children by distance again | 10.7 | 3.1 | 4.6 | 7.5 | 558 |
| `d18ccbb` | 2026-10-16 | rnpt-render: headless renderer, settings saved as JSON next to every render; NIRC timing counters | 10.6 | 3.1 | 4.6 | 7.5 | 561 |

## cluster-100k

| commit | date | subject | primary | diffuse | shadow | ortho | build ms |
|---|---|---|---:|---:|---:|---:|---:|
| `ab81978` | 2026-06-12 | add basic bvh | 2.6 | 0.8 | 0.6 † | 2.3 | 20 |
| `2ebe9f1` | 2026-06-13 | various improvements | 2.8 | 1.0 | 0.8 † | 2.6 | 20 |
| `25fe426` | 2026-06-13 | intersection simd optimization with wide | 3.5 | 1.2 | 1.1 † | 3.7 | 19 |
| `124a279` | 2026-06-13 | change the bvh algorithm | 3.7 | 1.3 | 1.1 † | 3.9 | 15 |
| `3477fb2` | 2026-06-13 | modified SAH, clustering instead of morton for SOA triangles grouping | 8.3 | 2.8 | 2.6 † | 8.6 | 24 |
| `db51ac1` | 2026-06-13 | cleanup comments, and magic numbers | 8.6 | 2.8 | 2.6 † | 8.8 | 24 |
| `b1fb0ba` | 2026-06-13 | tweaks | 8.6 | 2.8 | 2.6 † | 8.8 | 26 |
| `0d2606b` | 2026-06-13 | add textures support | 8.5 | 2.8 | 2.6 † | 8.8 | 27 |
| `0778479` | 2026-06-13 | fix warnings | 8.5 | 2.8 | 2.6 † | 8.8 | 26 |
| `ec4fa65` | 2026-06-14 | optimize bvh | 9.8 | 3.7 | 3.1 † | 9.6 | 26 |
| `73eb3ff` | 2026-06-14 | slight bvh optimisation | 11.2 | 4.0 | 3.4 † | 11.0 | 26 |
| `c0e2370` | 2026-06-14 | bvh8 attempt | 18.3 | 7.2 | 5.9 † | 16.5 | 27 |
| `51ab953` | 2026-06-14 | mix performance tweaks | 17.5 | 7.2 | 5.7 † | 14.8 | 27 |
| `59e8bde` | 2026-06-14 | shadow ray optimization | 17.5 | 7.2 | 7.5 | 14.8 | 27 |
| `65c2817` | 2026-06-15 | refactorings | 17.5 | 7.2 | 7.5 | 14.8 | 27 |
| `b99763c` | 2026-06-15 | emissive lights | 17.5 | 7.2 | 7.4 | 14.8 | 27 |
| `19f08c5` | 2026-06-15 | generic light list (emissive mesh + anlytic) | 17.4 | 7.1 | 7.4 | 14.7 | 27 |
| `f0d9350` | 2026-06-16 | cleanup, add comments | 17.4 | 7.1 | 7.4 | 14.7 | 27 |
| `d73438c` | 2026-06-16 | add MIS support | 17.6 | 7.2 | 7.3 | 14.8 | 27 |
| `3ce2ba2` | 2026-06-17 | add hdr sky support | 17.7 | 7.3 | 7.3 | 14.9 | 27 |
| `18cc184` | 2026-06-17 | add benchmarks | 17.7 | 7.3 | 7.3 | 14.9 | 27 |
| `befb868` | 2026-06-18 | try octant order lut to avoid sortir children in bvh traversal | 16.8 | 6.1 | 6.1 | 15.2 | 27 |
| `4a099c0` | 2026-06-18 | extract BVH into rnpt-bvh crate with clean Embree-like interface | 15.6 | 5.7 | 5.8 | 14.0 | 17 |
| `fb5869f` | 2026-06-19 | Cook Torrance/GGX and PBR materials support. | 15.6 | 5.7 | 5.8 | 14.0 | 17 |
| `e8f107d` | 2026-06-20 | glass support | 15.2 | 5.6 | 5.5 | 13.5 | 17 |
| `d17de5d` | 2026-06-23 | use FMA for ray intersection | 15.5 | 5.8 | 5.7 | 0.0 ✗ | 17 |
| `1c9d44d` | 2026-06-24 | true restir, fix regression in bvh intersection | 15.4 | 5.7 | 5.7 | 13.9 | 17 |
| `63042ea` | 2026-06-26 | cleanup | 15.5 | 5.7 | 5.7 | 13.8 | 17 |
| `c020143` | 2026-06-27 | first nirc version | 15.5 | 5.7 | 5.7 | 13.8 | 17 |
| `29cac5b` | 2026-07-16 | cleanup before going public | 15.5 | 5.7 | 5.7 | 13.8 | 17 |
| `75e6c3a` | 2026-07-16 | remove dead code | 15.4 | 5.7 | 5.7 | 13.9 | 17 |
| `cbd3127` | 2026-10-15 | bvh: revert the octant LUT (befb868), sort children by distance again | 16.0 | 6.7 | 7.0 | 13.4 | 18 |
| `d18ccbb` | 2026-10-16 | rnpt-render: headless renderer, settings saved as JSON next to every render; NIRC timing counters | 16.1 | 6.7 | 7.0 | 13.4 | 17 |

## cluster-1m

| commit | date | subject | primary | diffuse | shadow | ortho | build ms |
|---|---|---|---:|---:|---:|---:|---:|
| `ab81978` | 2026-06-12 | add basic bvh | 1.2 | 0.4 | 0.2 † | 1.1 | 252 |
| `2ebe9f1` | 2026-06-13 | various improvements | 1.0 | 0.6 | 0.4 † | 1.0 | 242 |
| `25fe426` | 2026-06-13 | intersection simd optimization with wide | 1.4 | 0.8 | 0.6 † | 1.6 | 230 |
| `124a279` | 2026-06-13 | change the bvh algorithm | 1.4 | 0.8 | 0.7 † | 1.6 | 168 |
| `3477fb2` | 2026-06-13 | modified SAH, clustering instead of morton for SOA triangles grouping | 3.9 | 2.0 | 1.8 † | 3.9 | 271 |
| `db51ac1` | 2026-06-13 | cleanup comments, and magic numbers | 4.0 | 2.1 | 1.9 † | 4.0 | 283 |
| `b1fb0ba` | 2026-06-13 | tweaks | 4.0 | 2.1 | 1.9 † | 4.0 | 297 |
| `0d2606b` | 2026-06-13 | add textures support | 4.0 | 2.1 | 1.9 † | 4.0 | 309 |
| `0778479` | 2026-06-13 | fix warnings | 4.0 | 2.1 | 1.9 † | 4.0 | 300 |
| `ec4fa65` | 2026-06-14 | optimize bvh | 4.3 | 2.7 | 2.1 † | 4.0 | 304 |
| `73eb3ff` | 2026-06-14 | slight bvh optimisation | 4.9 | 2.9 | 2.4 † | 4.7 | 301 |
| `c0e2370` | 2026-06-14 | bvh8 attempt | 8.0 | 5.2 | 4.0 † | 7.1 | 310 |
| `51ab953` | 2026-06-14 | mix performance tweaks | 7.4 | 5.1 | 3.8 † | 6.1 | 313 |
| `59e8bde` | 2026-06-14 | shadow ray optimization | 7.3 | 5.1 | 4.4 | 6.1 | 309 |
| `65c2817` | 2026-06-15 | refactorings | 7.3 | 5.1 | 4.4 | 6.1 | 313 |
| `b99763c` | 2026-06-15 | emissive lights | 7.3 | 5.2 | 4.4 | 6.1 | 313 |
| `19f08c5` | 2026-06-15 | generic light list (emissive mesh + anlytic) | 7.3 | 5.1 | 4.4 | 6.1 | 311 |
| `f0d9350` | 2026-06-16 | cleanup, add comments | 7.3 | 5.1 | 4.4 | 6.1 | 313 |
| `d73438c` | 2026-06-16 | add MIS support | 7.4 | 5.1 | 4.3 | 6.1 | 310 |
| `3ce2ba2` | 2026-06-17 | add hdr sky support | 7.4 | 5.2 | 4.3 | 6.1 | 314 |
| `18cc184` | 2026-06-17 | add benchmarks | 7.4 | 5.1 | 4.3 | 6.1 | 310 |
| `befb868` | 2026-06-18 | try octant order lut to avoid sortir children in bvh traversal | 7.2 | 4.5 | 4.0 | 5.7 | 311 |
| `4a099c0` | 2026-06-18 | extract BVH into rnpt-bvh crate with clean Embree-like interface | 6.9 | 4.3 | 3.9 | 5.5 | 186 |
| `fb5869f` | 2026-06-19 | Cook Torrance/GGX and PBR materials support. | 6.9 | 4.3 | 3.8 | 5.5 | 189 |
| `e8f107d` | 2026-06-20 | glass support | 6.8 | 4.2 | 3.7 | 5.3 | 191 |
| `d17de5d` | 2026-06-23 | use FMA for ray intersection | 7.0 | 4.3 | 3.8 | 0.0 ✗ | 189 |
| `1c9d44d` | 2026-06-24 | true restir, fix regression in bvh intersection | 6.9 | 4.2 | 3.8 | 5.4 | 191 |
| `63042ea` | 2026-06-26 | cleanup | 6.9 | 4.2 | 3.8 | 5.4 | 190 |
| `c020143` | 2026-06-27 | first nirc version | 6.9 | 4.2 | 3.8 | 5.4 | 192 |
| `29cac5b` | 2026-07-16 | cleanup before going public | 6.9 | 4.2 | 3.8 | 5.4 | 190 |
| `75e6c3a` | 2026-07-16 | remove dead code | 6.9 | 4.2 | 3.8 | 5.4 | 189 |
| `cbd3127` | 2026-10-15 | bvh: revert the octant LUT (befb868), sort children by distance again | 7.1 | 4.9 | 4.2 | 6.0 | 188 |
| `d18ccbb` | 2026-10-16 | rnpt-render: headless renderer, settings saved as JSON next to every render; NIRC timing counters | 7.1 | 4.9 | 4.2 | 5.9 | 188 |

## hf-100k

| commit | date | subject | primary | diffuse | shadow | ortho | build ms |
|---|---|---|---:|---:|---:|---:|---:|
| `ab81978` | 2026-06-12 | add basic bvh | 2.0 | 0.7 | 0.5 † | 1.2 | 21 |
| `2ebe9f1` | 2026-06-13 | various improvements | 2.2 | 0.8 | 0.7 † | 1.4 | 20 |
| `25fe426` | 2026-06-13 | intersection simd optimization with wide | 3.1 | 1.1 | 1.0 † | 2.1 | 19 |
| `124a279` | 2026-06-13 | change the bvh algorithm | 3.2 | 1.2 | 1.1 † | 2.3 | 14 |
| `3477fb2` | 2026-06-13 | modified SAH, clustering instead of morton for SOA triangles grouping | 6.8 | 2.6 | 2.4 † | 4.9 | 23 |
| `db51ac1` | 2026-06-13 | cleanup comments, and magic numbers | 7.0 | 2.7 | 2.5 † | 5.1 | 27 |
| `b1fb0ba` | 2026-06-13 | tweaks | 7.0 | 2.6 | 2.5 † | 5.1 | 25 |
| `0d2606b` | 2026-06-13 | add textures support | 6.9 | 2.6 | 2.5 † | 5.1 | 26 |
| `0778479` | 2026-06-13 | fix warnings | 6.9 | 2.6 | 2.5 † | 5.0 | 26 |
| `ec4fa65` | 2026-06-14 | optimize bvh | 7.8 | 3.3 | 2.8 † | 5.3 | 26 |
| `73eb3ff` | 2026-06-14 | slight bvh optimisation | 8.7 | 3.5 | 3.1 † | 5.9 | 26 |
| `c0e2370` | 2026-06-14 | bvh8 attempt | 13.7 | 6.5 | 5.3 † | 8.8 | 26 |
| `51ab953` | 2026-06-14 | mix performance tweaks | 12.7 | 6.5 | 5.1 † | 7.9 | 26 |
| `59e8bde` | 2026-06-14 | shadow ray optimization | 12.7 | 6.5 | 5.2 | 7.8 | 26 |
| `65c2817` | 2026-06-15 | refactorings | 12.7 | 6.5 | 5.2 | 7.8 | 26 |
| `b99763c` | 2026-06-15 | emissive lights | 12.7 | 6.4 | 5.2 | 7.9 | 26 |
| `19f08c5` | 2026-06-15 | generic light list (emissive mesh + anlytic) | 12.6 | 6.4 | 5.2 | 7.8 | 26 |
| `f0d9350` | 2026-06-16 | cleanup, add comments | 12.5 | 6.4 | 5.2 | 7.7 | 26 |
| `d73438c` | 2026-06-16 | add MIS support | 12.7 | 6.5 | 5.1 | 7.8 | 26 |
| `3ce2ba2` | 2026-06-17 | add hdr sky support | 12.8 | 6.5 | 5.1 | 7.9 | 26 |
| `18cc184` | 2026-06-17 | add benchmarks | 12.7 | 6.5 | 5.1 | 7.8 | 27 |
| `befb868` | 2026-06-18 | try octant order lut to avoid sortir children in bvh traversal | 10.5 | 5.4 | 5.4 | 7.6 | 26 |
| `4a099c0` | 2026-06-18 | extract BVH into rnpt-bvh crate with clean Embree-like interface | 10.0 | 5.1 | 5.2 | 7.4 | 17 |
| `fb5869f` | 2026-06-19 | Cook Torrance/GGX and PBR materials support. | 10.0 | 5.1 | 5.2 | 7.3 | 17 |
| `e8f107d` | 2026-06-20 | glass support | 9.7 | 5.0 | 5.0 | 7.1 | 17 |
| `d17de5d` | 2026-06-23 | use FMA for ray intersection | 10.0 | 5.1 | 5.2 | 0.0 | 17 |
| `1c9d44d` | 2026-06-24 | true restir, fix regression in bvh intersection | 9.9 | 5.1 | 5.1 | 7.2 | 17 |
| `63042ea` | 2026-06-26 | cleanup | 9.9 | 5.1 | 5.1 | 7.2 | 17 |
| `c020143` | 2026-06-27 | first nirc version | 9.9 | 5.1 | 5.1 | 7.2 | 17 |
| `29cac5b` | 2026-07-16 | cleanup before going public | 9.9 | 5.1 | 5.1 | 7.2 | 17 |
| `75e6c3a` | 2026-07-16 | remove dead code | 9.9 | 5.1 | 5.1 | 7.2 | 17 |
| `cbd3127` | 2026-10-15 | bvh: revert the octant LUT (befb868), sort children by distance again | 11.8 | 6.1 | 5.0 | 7.2 | 18 |
| `d18ccbb` | 2026-10-16 | rnpt-render: headless renderer, settings saved as JSON next to every render; NIRC timing counters | 11.8 | 6.1 | 5.0 | 7.2 | 17 |

## hf-1m

| commit | date | subject | primary | diffuse | shadow | ortho | build ms |
|---|---|---|---:|---:|---:|---:|---:|
| `ab81978` | 2026-06-12 | add basic bvh | 1.4 | 0.4 | 0.3 † | 0.8 | 240 |
| `2ebe9f1` | 2026-06-13 | various improvements | 1.4 | 0.5 | 0.4 † | 0.8 | 239 |
| `25fe426` | 2026-06-13 | intersection simd optimization with wide | 2.0 | 0.7 | 0.6 † | 1.2 | 218 |
| `124a279` | 2026-06-13 | change the bvh algorithm | 2.0 | 0.7 | 0.6 † | 1.2 | 160 |
| `3477fb2` | 2026-06-13 | modified SAH, clustering instead of morton for SOA triangles grouping | 5.2 | 1.9 | 1.6 † | 2.7 | 263 |
| `db51ac1` | 2026-06-13 | cleanup comments, and magic numbers | 5.3 | 2.0 | 1.6 † | 2.7 | 267 |
| `b1fb0ba` | 2026-06-13 | tweaks | 5.3 | 2.0 | 1.6 † | 2.7 | 284 |
| `0d2606b` | 2026-06-13 | add textures support | 5.2 | 1.9 | 1.6 † | 2.7 | 295 |
| `0778479` | 2026-06-13 | fix warnings | 5.2 | 1.9 | 1.6 † | 2.7 | 290 |
| `ec4fa65` | 2026-06-14 | optimize bvh | 6.1 | 2.4 | 1.8 † | 3.0 | 293 |
| `73eb3ff` | 2026-06-14 | slight bvh optimisation | 6.8 | 2.6 | 2.0 † | 3.2 | 290 |
| `c0e2370` | 2026-06-14 | bvh8 attempt | 11.2 | 4.9 | 3.5 † | 4.4 | 295 |
| `51ab953` | 2026-06-14 | mix performance tweaks | 10.6 | 5.0 | 3.4 † | 4.3 | 297 |
| `59e8bde` | 2026-06-14 | shadow ray optimization | 10.5 | 4.9 | 3.3 | 4.3 | 296 |
| `65c2817` | 2026-06-15 | refactorings | 10.5 | 5.0 | 3.3 | 4.2 | 298 |
| `b99763c` | 2026-06-15 | emissive lights | 10.6 | 4.9 | 3.3 | 4.2 | 297 |
| `19f08c5` | 2026-06-15 | generic light list (emissive mesh + anlytic) | 10.4 | 4.9 | 3.2 | 4.2 | 298 |
| `f0d9350` | 2026-06-16 | cleanup, add comments | 10.5 | 4.9 | 3.2 | 4.3 | 297 |
| `d73438c` | 2026-06-16 | add MIS support | 10.6 | 4.9 | 3.2 | 4.4 | 308 |
| `3ce2ba2` | 2026-06-17 | add hdr sky support | 10.6 | 5.0 | 3.2 | 4.4 | 301 |
| `18cc184` | 2026-06-17 | add benchmarks | 10.5 | 4.9 | 3.2 | 4.4 | 301 |
| `befb868` | 2026-06-18 | try octant order lut to avoid sortir children in bvh traversal | 10.1 | 4.1 | 3.5 | 4.3 | 304 |
| `4a099c0` | 2026-06-18 | extract BVH into rnpt-bvh crate with clean Embree-like interface | 9.7 | 3.9 | 3.4 | 4.2 | 181 |
| `fb5869f` | 2026-06-19 | Cook Torrance/GGX and PBR materials support. | 9.7 | 4.0 | 3.4 | 4.3 | 181 |
| `e8f107d` | 2026-06-20 | glass support | 9.6 | 3.8 | 3.3 | 4.2 | 184 |
| `d17de5d` | 2026-06-23 | use FMA for ray intersection | 9.7 | 4.0 | 3.4 | 0.0 | 185 |
| `1c9d44d` | 2026-06-24 | true restir, fix regression in bvh intersection | 9.5 | 3.9 | 3.4 | 4.2 | 186 |
| `63042ea` | 2026-06-26 | cleanup | 9.5 | 3.9 | 3.4 | 4.2 | 187 |
| `c020143` | 2026-06-27 | first nirc version | 9.6 | 3.9 | 3.4 | 4.2 | 186 |
| `29cac5b` | 2026-07-16 | cleanup before going public | 9.5 | 3.9 | 3.4 | 4.2 | 186 |
| `75e6c3a` | 2026-07-16 | remove dead code | 9.5 | 3.9 | 3.4 | 4.2 | 186 |
| `cbd3127` | 2026-10-15 | bvh: revert the octant LUT (befb868), sort children by distance again | 9.9 | 4.6 | 3.1 | 4.3 | 185 |
| `d18ccbb` | 2026-10-16 | rnpt-render: headless renderer, settings saved as JSON next to every render; NIRC timing counters | 9.7 | 4.6 | 3.1 | 4.2 | 184 |

## sponza

| commit | date | subject | primary | diffuse | shadow | ortho | build ms |
|---|---|---|---:|---:|---:|---:|---:|
| `ab81978` | 2026-06-12 | add basic bvh | 1.8 | 0.9 | 0.7 † | 0.9 | 63 |
| `2ebe9f1` | 2026-06-13 | various improvements | 2.0 | 1.0 | 0.8 † | 1.1 | 63 |
| `25fe426` | 2026-06-13 | intersection simd optimization with wide | 3.1 | 1.7 | 1.4 † | 1.6 | 59 |
| `124a279` | 2026-06-13 | change the bvh algorithm | 3.1 | 1.7 | 1.4 † | 1.6 | 46 |
| `3477fb2` | 2026-06-13 | modified SAH, clustering instead of morton for SOA triangles grouping | 6.9 | 3.8 | 2.9 † | 4.2 | 72 |
| `db51ac1` | 2026-06-13 | cleanup comments, and magic numbers | 7.4 | 4.1 | 3.3 † | 4.8 | 72 |
| `b1fb0ba` | 2026-06-13 | tweaks | 7.4 | 4.1 | 3.3 † | 4.8 | 77 |
| `0d2606b` | 2026-06-13 | add textures support | 7.4 | 4.1 | 3.3 † | 4.8 | 81 |
| `0778479` | 2026-06-13 | fix warnings | 7.4 | 4.1 | 3.3 † | 4.8 | 79 |
| `ec4fa65` | 2026-06-14 | optimize bvh | 7.5 | 4.9 | 3.6 † | 4.5 | 79 |
| `73eb3ff` | 2026-06-14 | slight bvh optimisation | 9.3 | 5.6 | 4.3 † | 5.9 | 79 |
| `c0e2370` | 2026-06-14 | bvh8 attempt | 13.1 | 8.8 | 8.2 † | 8.5 | 81 |
| `51ab953` | 2026-06-14 | mix performance tweaks | 12.3 | 8.9 | 8.1 † | 7.7 | 80 |
| `59e8bde` | 2026-06-14 | shadow ray optimization | 12.3 | 9.0 | 10.7 | 7.6 | 81 |
| `65c2817` | 2026-06-15 | refactorings | 12.3 | 8.9 | 10.6 | 7.7 | 80 |
| `b99763c` | 2026-06-15 | emissive lights | 12.3 | 9.0 | 10.6 | 7.7 | 80 |
| `19f08c5` | 2026-06-15 | generic light list (emissive mesh + anlytic) | 12.2 | 8.9 | 10.5 | 7.6 | 80 |
| `f0d9350` | 2026-06-16 | cleanup, add comments | 12.3 | 8.9 | 10.6 | 7.6 | 80 |
| `d73438c` | 2026-06-16 | add MIS support | 12.3 | 9.0 | 10.5 | 7.6 | 81 |
| `3ce2ba2` | 2026-06-17 | add hdr sky support | 12.3 | 9.0 | 10.5 | 7.6 | 81 |
| `18cc184` | 2026-06-17 | add benchmarks | 12.3 | 9.0 | 10.5 | 7.6 | 82 |
| `befb868` | 2026-06-18 | try octant order lut to avoid sortir children in bvh traversal | 10.2 | 8.2 | 7.3 | 8.6 | 81 |
| `4a099c0` | 2026-06-18 | extract BVH into rnpt-bvh crate with clean Embree-like interface | 9.6 | 7.6 | 6.9 | 8.1 | 48 |
| `fb5869f` | 2026-06-19 | Cook Torrance/GGX and PBR materials support. | 9.6 | 7.6 | 6.9 | 8.1 | 48 |
| `e8f107d` | 2026-06-20 | glass support | 9.1 | 7.2 | 6.4 | 7.4 | 49 |
| `d17de5d` | 2026-06-23 | use FMA for ray intersection | 9.6 | 7.6 | 6.8 | 0.0 ✗ | 49 |
| `1c9d44d` | 2026-06-24 | true restir, fix regression in bvh intersection | 9.5 | 7.6 | 6.7 | 7.9 | 49 |
| `63042ea` | 2026-06-26 | cleanup | 9.5 | 7.6 | 6.7 | 7.9 | 49 |
| `c020143` | 2026-06-27 | first nirc version | 9.6 | 7.5 | 6.7 | 7.9 | 49 |
| `29cac5b` | 2026-07-16 | cleanup before going public | 9.5 | 7.5 | 6.7 | 7.9 | 51 |
| `75e6c3a` | 2026-07-16 | remove dead code | 9.6 | 7.5 | 6.7 | 7.9 | 49 |
| `cbd3127` | 2026-10-15 | bvh: revert the octant LUT (befb868), sort children by distance again | 11.2 | 8.2 | 9.5 | 7.1 | 49 |
| `d18ccbb` | 2026-10-16 | rnpt-render: headless renderer, settings saved as JSON next to every render; NIRC timing counters | 11.2 | 8.2 | 9.5 | 7.1 | 48 |
