# BVH history — amd-ryzen-7-9700x-8-core — build config: head

Single-thread throughput in Mrays/s: median over processes of the median pass. ✗ = results disagree with the brute-force reference; † = no occlusion query yet, shadow rays use closest-hit.

## bistro-ext

| commit | date | subject | primary | diffuse | shadow | ortho | build ms |
|---|---|---|---:|---:|---:|---:|---:|
| `ab81978` | 2026-06-12 | add basic bvh | 1.5 | 0.3 | 0.2 † | 0.8 | 943 |
| `2ebe9f1` | 2026-06-13 | various improvements | 1.1 | 0.3 | 0.3 † | 0.7 | 937 |
| `25fe426` | 2026-06-13 | intersection simd optimization with wide | 1.4 | 0.4 | 0.4 † | 1.0 | 935 |
| `124a279` | 2026-06-13 | change the bvh algorithm | 1.6 | 0.4 | 0.4 † | 1.1 | 732 |
| `3477fb2` | 2026-06-13 | modified SAH, clustering instead of morton for SOA triangles grouping | 5.7 | 1.4 | 1.5 † | 4.6 | 1025 |
| `db51ac1` | 2026-06-13 | cleanup comments, and magic numbers | 5.9 | 1.4 | 1.5 † | 4.7 | 1062 |
| `b1fb0ba` | 2026-06-13 | tweaks | 5.9 | 1.5 | 1.5 † | 4.6 | 1097 |
| `0d2606b` | 2026-06-13 | add textures support | 5.9 | 1.4 | 1.5 † | 4.7 | 1122 |
| `0778479` | 2026-06-13 | fix warnings | 5.9 | 1.4 | 1.5 † | 4.7 | 1140 |
| `ec4fa65` | 2026-06-14 | optimize bvh | 6.4 | 1.6 | 1.7 † | 4.9 | 1119 |
| `73eb3ff` | 2026-06-14 | slight bvh optimisation | 7.4 | 1.8 | 1.9 † | 5.7 | 1128 |
| `c0e2370` | 2026-06-14 | bvh8 attempt | 11.6 | 3.1 | 3.4 † | 8.2 | 1133 |
| `51ab953` | 2026-06-14 | mix performance tweaks | 11.0 | 3.1 | 3.3 † | 7.8 | 1138 |
| `59e8bde` | 2026-06-14 | shadow ray optimization | 11.0 | 3.1 | 4.8 | 7.8 | 1141 |
| `65c2817` | 2026-06-15 | refactorings | 11.0 | 3.1 | 4.8 | 7.8 | 1157 |
| `b99763c` | 2026-06-15 | emissive lights | 11.1 | 3.1 | 4.8 | 7.8 | 1134 |
| `19f08c5` | 2026-06-15 | generic light list (emissive mesh + anlytic) | 10.9 | 3.1 | 4.8 | 7.8 | 1160 |
| `f0d9350` | 2026-06-16 | cleanup, add comments | 11.0 | 3.1 | 4.8 | 7.8 | 1137 |
| `d73438c` | 2026-06-16 | add MIS support | 11.1 | 3.1 | 4.7 | 7.9 | 1158 |
| `3ce2ba2` | 2026-06-17 | add hdr sky support | 11.1 | 3.1 | 4.7 | 7.8 | 1168 |
| `18cc184` | 2026-06-17 | add benchmarks | 10.9 | 3.0 | 4.6 | 7.7 | 1174 |
| `befb868` | 2026-06-18 | try octant order lut to avoid sortir children in bvh traversal | 10.1 | 2.7 | 4.5 | 7.6 | 1179 |
| `4a099c0` | 2026-06-18 | extract BVH into rnpt-bvh crate with clean Embree-like interface | 9.7 | 2.7 | 4.4 | 7.4 | 553 |
| `fb5869f` | 2026-06-19 | Cook Torrance/GGX and PBR materials support. | 9.7 | 2.6 | 4.3 | 7.4 | 555 |
| `e8f107d` | 2026-06-20 | glass support | 9.4 | 2.5 | 4.1 | 6.9 | 561 |
| `d17de5d` | 2026-06-23 | use FMA for ray intersection | 9.8 | 2.7 | 4.3 | 0.0 | 560 |
| `1c9d44d` | 2026-06-24 | true restir, fix regression in bvh intersection | 9.8 | 2.7 | 4.3 | 7.3 | 568 |
| `63042ea` | 2026-06-26 | cleanup | 9.7 | 2.7 | 4.3 | 7.4 | 564 |
| `c020143` | 2026-06-27 | first nirc version | 9.8 | 2.7 | 4.3 | 7.4 | 565 |
| `29cac5b` | 2026-07-16 | cleanup before going public | 9.7 | 2.7 | 4.3 | 7.3 | 564 |
| `75e6c3a` | 2026-07-16 | remove dead code | 9.7 | 2.7 | 4.3 | 7.4 | 563 |
| `cbd3127` | 2026-10-15 | bvh: revert the octant LUT (befb868), sort children by distance again | 10.6 | 3.1 | 4.6 | 7.5 | 562 |
| `d18ccbb` | 2026-10-16 | rnpt-render: headless renderer, settings saved as JSON next to every render; NIRC timing counters | 10.6 | 3.1 | 4.6 | 7.5 | 561 |

## cluster-100k

| commit | date | subject | primary | diffuse | shadow | ortho | build ms |
|---|---|---|---:|---:|---:|---:|---:|
| `ab81978` | 2026-06-12 | add basic bvh | 3.1 | 0.9 | 0.7 † | 2.9 | 19 |
| `2ebe9f1` | 2026-06-13 | various improvements | 2.8 | 1.0 | 0.8 † | 2.8 | 18 |
| `25fe426` | 2026-06-13 | intersection simd optimization with wide | 3.6 | 1.3 | 1.1 † | 3.7 | 19 |
| `124a279` | 2026-06-13 | change the bvh algorithm | 3.8 | 1.3 | 1.1 † | 3.9 | 14 |
| `3477fb2` | 2026-06-13 | modified SAH, clustering instead of morton for SOA triangles grouping | 8.4 | 2.8 | 2.6 † | 8.7 | 23 |
| `db51ac1` | 2026-06-13 | cleanup comments, and magic numbers | 8.6 | 2.8 | 2.6 † | 8.9 | 23 |
| `b1fb0ba` | 2026-06-13 | tweaks | 8.6 | 2.8 | 2.6 † | 8.9 | 25 |
| `0d2606b` | 2026-06-13 | add textures support | 8.6 | 2.8 | 2.6 † | 8.9 | 26 |
| `0778479` | 2026-06-13 | fix warnings | 8.5 | 2.8 | 2.6 † | 8.9 | 26 |
| `ec4fa65` | 2026-06-14 | optimize bvh | 9.8 | 3.7 | 3.1 † | 9.6 | 26 |
| `73eb3ff` | 2026-06-14 | slight bvh optimisation | 11.4 | 4.0 | 3.5 † | 11.3 | 26 |
| `c0e2370` | 2026-06-14 | bvh8 attempt | 18.3 | 7.2 | 5.9 † | 16.5 | 26 |
| `51ab953` | 2026-06-14 | mix performance tweaks | 17.6 | 7.3 | 5.7 † | 14.8 | 26 |
| `59e8bde` | 2026-06-14 | shadow ray optimization | 17.6 | 7.2 | 7.5 | 14.8 | 26 |
| `65c2817` | 2026-06-15 | refactorings | 17.6 | 7.3 | 7.5 | 14.9 | 26 |
| `b99763c` | 2026-06-15 | emissive lights | 17.7 | 7.2 | 7.4 | 14.9 | 26 |
| `19f08c5` | 2026-06-15 | generic light list (emissive mesh + anlytic) | 17.4 | 7.2 | 7.4 | 14.8 | 27 |
| `f0d9350` | 2026-06-16 | cleanup, add comments | 17.5 | 7.2 | 7.4 | 14.8 | 26 |
| `d73438c` | 2026-06-16 | add MIS support | 17.7 | 7.3 | 7.3 | 14.8 | 27 |
| `3ce2ba2` | 2026-06-17 | add hdr sky support | 17.7 | 7.3 | 7.3 | 14.9 | 26 |
| `18cc184` | 2026-06-17 | add benchmarks | 17.7 | 7.3 | 7.3 | 14.9 | 27 |
| `befb868` | 2026-06-18 | try octant order lut to avoid sortir children in bvh traversal | 16.9 | 6.1 | 6.1 | 15.2 | 27 |
| `4a099c0` | 2026-06-18 | extract BVH into rnpt-bvh crate with clean Embree-like interface | 15.7 | 5.7 | 5.8 | 14.0 | 17 |
| `fb5869f` | 2026-06-19 | Cook Torrance/GGX and PBR materials support. | 15.6 | 5.7 | 5.8 | 14.0 | 17 |
| `e8f107d` | 2026-06-20 | glass support | 15.1 | 5.5 | 5.5 | 13.5 | 17 |
| `d17de5d` | 2026-06-23 | use FMA for ray intersection | 15.5 | 5.7 | 5.8 | 0.0 ✗ | 17 |
| `1c9d44d` | 2026-06-24 | true restir, fix regression in bvh intersection | 15.4 | 5.7 | 5.7 | 13.8 | 17 |
| `63042ea` | 2026-06-26 | cleanup | 15.5 | 5.7 | 5.7 | 13.9 | 17 |
| `c020143` | 2026-06-27 | first nirc version | 15.5 | 5.7 | 5.7 | 13.8 | 17 |
| `29cac5b` | 2026-07-16 | cleanup before going public | 15.5 | 5.6 | 5.7 | 13.9 | 17 |
| `75e6c3a` | 2026-07-16 | remove dead code | 15.5 | 5.7 | 5.7 | 13.8 | 17 |
| `cbd3127` | 2026-10-15 | bvh: revert the octant LUT (befb868), sort children by distance again | 16.1 | 6.7 | 7.0 | 13.4 | 17 |
| `d18ccbb` | 2026-10-16 | rnpt-render: headless renderer, settings saved as JSON next to every render; NIRC timing counters | 16.0 | 6.7 | 7.0 | 13.4 | 17 |
| `0a05f36` | 2026-10-15 | bvh step 14: back to (p - o) * inv_dir in the box test, fixes axis-aligned rays (1c9d44d) | 15.2 | 5.6 | 5.6 | 13.6 | 17 |
| `14da82f` | 2026-10-15 | bvh step 15 (experiment): closest hit sorts children again, the octant LUT stays unused | 15.9 | 6.3 | 5.7 | 13.2 | 17 |
| `379ff20` | 2026-10-15 | bvh step 7: traversal stack without bounds checks (73eb3ff) | 9.8 | 3.7 | 3.2 | 9.6 | 17 |
| `4ea22b1` | 2026-10-15 | bvh step 1: front-to-back traversal, prune nodes behind the closest hit (2ebe9f1) | 2.8 | 1.0 | 0.8 | 2.7 | 10 |
| `797a814` | 2026-10-15 | bvh step 17: revert the octant LUT (befb868): sort children by distance again, slots in build order | 15.7 | 6.6 | 6.8 | 13.2 | 17 |
| `867fb20` | 2026-10-15 | bvh step 0: binary BVH, Morton order, median split (ab81978) | 3.0 | 0.9 | 0.7 | 2.8 | 10 |
| `96c3a79` | 2026-10-15 | bvh step 5: 16 SAH bins instead of 8 (db51ac1, b1fb0ba) | 7.8 | 2.5 | 2.4 | 8.2 | 17 |
| `ae1542e` | 2026-10-15 | bvh step 3: pack 8 consecutive triangles into chunks first, build the tree over chunks (124a279) | 3.6 | 1.3 | 1.1 | 3.7 | 6 |
| `c0ce714` | 2026-10-15 | bvh step 12: octant order LUT instead of sorting children, octant slots in the builder (befb868) | 14.8 | 5.5 | 5.5 | 13.2 | 17 |
| `c1a0715` | 2026-10-15 | bvh step 4: cluster triangles into chunks by median splits, binned SAH over chunks (3477fb2) | 7.8 | 2.7 | 2.5 | 8.2 | 15 |
| `c433e1b` | 2026-10-15 | bvh step 16 (experiment): any-hit visits children near-first using the octant LUT | 15.9 | 6.3 | 5.6 | 13.1 | 17 |
| `dee98f2` | 2026-10-15 | bvh step 11: final code layout (leaf encoding helpers, TriSlot, per-chunk double-sided masks), same algorithm (65c2817, 4a099c0, e8f107d) | 15.5 | 6.5 | 6.8 | 13.0 | 17 |
| `e27585e` | 2026-10-15 | bvh step 6: precomputed reciprocal direction, branchless slab test, unchecked node access (ec4fa65) | 8.6 | 3.4 | 2.9 | 8.4 | 17 |
| `f0126d9` | 2026-10-15 | bvh step 2: 8-triangle SoA leaves, SIMD intersection with wide (25fe426) | 3.4 | 1.2 | 1.1 | 3.5 | 10 |

## cluster-1m

| commit | date | subject | primary | diffuse | shadow | ortho | build ms |
|---|---|---|---:|---:|---:|---:|---:|
| `ab81978` | 2026-06-12 | add basic bvh | 1.4 | 0.5 | 0.3 † | 1.3 | 227 |
| `2ebe9f1` | 2026-06-13 | various improvements | 1.0 | 0.6 | 0.4 † | 1.0 | 228 |
| `25fe426` | 2026-06-13 | intersection simd optimization with wide | 1.4 | 0.8 | 0.6 † | 1.6 | 223 |
| `124a279` | 2026-06-13 | change the bvh algorithm | 1.4 | 0.8 | 0.7 † | 1.6 | 165 |
| `3477fb2` | 2026-06-13 | modified SAH, clustering instead of morton for SOA triangles grouping | 3.9 | 2.0 | 1.8 † | 3.9 | 273 |
| `db51ac1` | 2026-06-13 | cleanup comments, and magic numbers | 4.0 | 2.1 | 1.9 † | 4.0 | 275 |
| `b1fb0ba` | 2026-06-13 | tweaks | 4.0 | 2.1 | 1.9 † | 4.0 | 290 |
| `0d2606b` | 2026-06-13 | add textures support | 4.0 | 2.1 | 1.9 † | 4.0 | 301 |
| `0778479` | 2026-06-13 | fix warnings | 4.0 | 2.1 | 1.9 † | 4.0 | 300 |
| `ec4fa65` | 2026-06-14 | optimize bvh | 4.3 | 2.7 | 2.1 † | 4.0 | 301 |
| `73eb3ff` | 2026-06-14 | slight bvh optimisation | 5.0 | 2.9 | 2.4 † | 4.7 | 300 |
| `c0e2370` | 2026-06-14 | bvh8 attempt | 8.0 | 5.1 | 4.0 † | 7.1 | 308 |
| `51ab953` | 2026-06-14 | mix performance tweaks | 7.3 | 5.1 | 3.8 † | 6.1 | 305 |
| `59e8bde` | 2026-06-14 | shadow ray optimization | 7.3 | 5.1 | 4.4 | 6.1 | 309 |
| `65c2817` | 2026-06-15 | refactorings | 7.3 | 5.1 | 4.3 | 6.0 | 308 |
| `b99763c` | 2026-06-15 | emissive lights | 7.3 | 5.0 | 4.3 | 6.1 | 306 |
| `19f08c5` | 2026-06-15 | generic light list (emissive mesh + anlytic) | 7.3 | 5.0 | 4.3 | 6.0 | 306 |
| `f0d9350` | 2026-06-16 | cleanup, add comments | 7.3 | 5.1 | 4.4 | 6.1 | 307 |
| `d73438c` | 2026-06-16 | add MIS support | 7.3 | 5.1 | 4.3 | 6.1 | 310 |
| `3ce2ba2` | 2026-06-17 | add hdr sky support | 7.3 | 5.1 | 4.3 | 6.1 | 309 |
| `18cc184` | 2026-06-17 | add benchmarks | 7.4 | 5.1 | 4.3 | 6.1 | 315 |
| `befb868` | 2026-06-18 | try octant order lut to avoid sortir children in bvh traversal | 7.2 | 4.4 | 3.9 | 5.6 | 312 |
| `4a099c0` | 2026-06-18 | extract BVH into rnpt-bvh crate with clean Embree-like interface | 6.9 | 4.2 | 3.8 | 5.4 | 185 |
| `fb5869f` | 2026-06-19 | Cook Torrance/GGX and PBR materials support. | 6.9 | 4.2 | 3.8 | 5.5 | 186 |
| `e8f107d` | 2026-06-20 | glass support | 6.7 | 4.1 | 3.7 | 5.3 | 188 |
| `d17de5d` | 2026-06-23 | use FMA for ray intersection | 7.0 | 4.3 | 3.8 | 0.0 ✗ | 188 |
| `1c9d44d` | 2026-06-24 | true restir, fix regression in bvh intersection | 6.9 | 4.2 | 3.8 | 5.4 | 189 |
| `63042ea` | 2026-06-26 | cleanup | 6.9 | 4.2 | 3.8 | 5.4 | 188 |
| `c020143` | 2026-06-27 | first nirc version | 6.9 | 4.2 | 3.8 | 5.4 | 191 |
| `29cac5b` | 2026-07-16 | cleanup before going public | 6.9 | 4.2 | 3.8 | 5.5 | 188 |
| `75e6c3a` | 2026-07-16 | remove dead code | 6.9 | 4.2 | 3.8 | 5.5 | 189 |
| `cbd3127` | 2026-10-15 | bvh: revert the octant LUT (befb868), sort children by distance again | 7.1 | 4.9 | 4.2 | 5.9 | 189 |
| `d18ccbb` | 2026-10-16 | rnpt-render: headless renderer, settings saved as JSON next to every render; NIRC timing counters | 7.1 | 4.9 | 4.2 | 5.9 | 189 |

## hf-100k

| commit | date | subject | primary | diffuse | shadow | ortho | build ms |
|---|---|---|---:|---:|---:|---:|---:|
| `ab81978` | 2026-06-12 | add basic bvh | 2.4 | 0.8 | 0.5 † | 1.5 | 19 |
| `2ebe9f1` | 2026-06-13 | various improvements | 2.2 | 0.8 | 0.7 † | 1.5 | 19 |
| `25fe426` | 2026-06-13 | intersection simd optimization with wide | 3.1 | 1.1 | 1.0 † | 2.1 | 18 |
| `124a279` | 2026-06-13 | change the bvh algorithm | 3.2 | 1.2 | 1.1 † | 2.3 | 14 |
| `3477fb2` | 2026-06-13 | modified SAH, clustering instead of morton for SOA triangles grouping | 6.8 | 2.6 | 2.4 † | 5.0 | 23 |
| `db51ac1` | 2026-06-13 | cleanup comments, and magic numbers | 7.0 | 2.7 | 2.5 † | 5.1 | 23 |
| `b1fb0ba` | 2026-06-13 | tweaks | 7.0 | 2.6 | 2.5 † | 5.1 | 25 |
| `0d2606b` | 2026-06-13 | add textures support | 7.0 | 2.6 | 2.5 † | 5.1 | 25 |
| `0778479` | 2026-06-13 | fix warnings | 7.0 | 2.6 | 2.5 † | 5.1 | 26 |
| `ec4fa65` | 2026-06-14 | optimize bvh | 7.8 | 3.3 | 2.8 † | 5.3 | 25 |
| `73eb3ff` | 2026-06-14 | slight bvh optimisation | 8.8 | 3.5 | 3.1 † | 6.0 | 25 |
| `c0e2370` | 2026-06-14 | bvh8 attempt | 13.8 | 6.5 | 5.3 † | 8.9 | 26 |
| `51ab953` | 2026-06-14 | mix performance tweaks | 12.7 | 6.5 | 5.1 † | 7.8 | 26 |
| `59e8bde` | 2026-06-14 | shadow ray optimization | 12.7 | 6.4 | 5.2 | 7.8 | 26 |
| `65c2817` | 2026-06-15 | refactorings | 12.7 | 6.5 | 5.3 | 7.9 | 26 |
| `b99763c` | 2026-06-15 | emissive lights | 12.8 | 6.4 | 5.2 | 7.9 | 26 |
| `19f08c5` | 2026-06-15 | generic light list (emissive mesh + anlytic) | 12.6 | 6.4 | 5.2 | 7.8 | 26 |
| `f0d9350` | 2026-06-16 | cleanup, add comments | 12.6 | 6.4 | 5.2 | 7.8 | 26 |
| `d73438c` | 2026-06-16 | add MIS support | 12.7 | 6.5 | 5.1 | 7.8 | 26 |
| `3ce2ba2` | 2026-06-17 | add hdr sky support | 12.7 | 6.5 | 5.1 | 7.8 | 26 |
| `18cc184` | 2026-06-17 | add benchmarks | 12.8 | 6.5 | 5.1 | 7.9 | 26 |
| `befb868` | 2026-06-18 | try octant order lut to avoid sortir children in bvh traversal | 10.5 | 5.4 | 5.4 | 7.7 | 26 |
| `4a099c0` | 2026-06-18 | extract BVH into rnpt-bvh crate with clean Embree-like interface | 10.0 | 5.1 | 5.2 | 7.4 | 17 |
| `fb5869f` | 2026-06-19 | Cook Torrance/GGX and PBR materials support. | 10.0 | 5.1 | 5.2 | 7.3 | 17 |
| `e8f107d` | 2026-06-20 | glass support | 9.7 | 5.0 | 5.0 | 7.1 | 17 |
| `d17de5d` | 2026-06-23 | use FMA for ray intersection | 10.0 | 5.2 | 5.2 | 0.0 | 17 |
| `1c9d44d` | 2026-06-24 | true restir, fix regression in bvh intersection | 9.9 | 5.1 | 5.1 | 7.2 | 17 |
| `63042ea` | 2026-06-26 | cleanup | 9.9 | 5.1 | 5.1 | 7.2 | 17 |
| `c020143` | 2026-06-27 | first nirc version | 9.9 | 5.1 | 5.1 | 7.2 | 17 |
| `29cac5b` | 2026-07-16 | cleanup before going public | 9.9 | 5.1 | 5.1 | 7.2 | 17 |
| `75e6c3a` | 2026-07-16 | remove dead code | 9.9 | 5.1 | 5.1 | 7.2 | 17 |
| `cbd3127` | 2026-10-15 | bvh: revert the octant LUT (befb868), sort children by distance again | 11.8 | 6.1 | 5.0 | 7.2 | 17 |
| `d18ccbb` | 2026-10-16 | rnpt-render: headless renderer, settings saved as JSON next to every render; NIRC timing counters | 11.8 | 6.1 | 5.0 | 7.2 | 17 |
| `0a05f36` | 2026-10-15 | bvh step 14: back to (p - o) * inv_dir in the box test, fixes axis-aligned rays (1c9d44d) | 9.7 | 5.0 | 5.0 | 7.1 | 17 |
| `14da82f` | 2026-10-15 | bvh step 15 (experiment): closest hit sorts children again, the octant LUT stays unused | 11.6 | 5.7 | 5.1 | 6.9 | 17 |
| `379ff20` | 2026-10-15 | bvh step 7: traversal stack without bounds checks (73eb3ff) | 7.9 | 3.3 | 2.9 | 5.2 | 17 |
| `4ea22b1` | 2026-10-15 | bvh step 1: front-to-back traversal, prune nodes behind the closest hit (2ebe9f1) | 2.2 | 0.8 | 0.7 | 1.5 | 10 |
| `797a814` | 2026-10-15 | bvh step 17: revert the octant LUT (befb868): sort children by distance again, slots in build order | 11.6 | 6.0 | 4.9 | 7.0 | 17 |
| `867fb20` | 2026-10-15 | bvh step 0: binary BVH, Morton order, median split (ab81978) | 2.3 | 0.8 | 0.5 | 1.4 | 10 |
| `96c3a79` | 2026-10-15 | bvh step 5: 16 SAH bins instead of 8 (db51ac1, b1fb0ba) | 6.3 | 2.4 | 2.3 | 4.5 | 18 |
| `ae1542e` | 2026-10-15 | bvh step 3: pack 8 consecutive triangles into chunks first, build the tree over chunks (124a279) | 3.1 | 1.1 | 1.0 | 2.2 | 6 |
| `c0ce714` | 2026-10-15 | bvh step 12: octant order LUT instead of sorting children, octant slots in the builder (befb868) | 9.5 | 4.9 | 4.9 | 7.0 | 18 |
| `c1a0715` | 2026-10-15 | bvh step 4: cluster triangles into chunks by median splits, binned SAH over chunks (3477fb2) | 6.4 | 2.5 | 2.3 | 4.6 | 15 |
| `c433e1b` | 2026-10-15 | bvh step 16 (experiment): any-hit visits children near-first using the octant LUT | 11.5 | 5.7 | 4.8 | 6.8 | 17 |
| `dee98f2` | 2026-10-15 | bvh step 11: final code layout (leaf encoding helpers, TriSlot, per-chunk double-sided masks), same algorithm (65c2817, 4a099c0, e8f107d) | 11.5 | 5.9 | 4.8 | 6.9 | 17 |
| `e27585e` | 2026-10-15 | bvh step 6: precomputed reciprocal direction, branchless slab test, unchecked node access (ec4fa65) | 7.0 | 3.0 | 2.6 | 4.6 | 18 |
| `f0126d9` | 2026-10-15 | bvh step 2: 8-triangle SoA leaves, SIMD intersection with wide (25fe426) | 3.0 | 1.1 | 1.0 | 2.0 | 10 |

## hf-1m

| commit | date | subject | primary | diffuse | shadow | ortho | build ms |
|---|---|---|---:|---:|---:|---:|---:|
| `ab81978` | 2026-06-12 | add basic bvh | 1.6 | 0.5 | 0.3 † | 0.9 | 219 |
| `2ebe9f1` | 2026-06-13 | various improvements | 1.5 | 0.6 | 0.4 † | 0.9 | 219 |
| `25fe426` | 2026-06-13 | intersection simd optimization with wide | 2.0 | 0.7 | 0.6 † | 1.2 | 213 |
| `124a279` | 2026-06-13 | change the bvh algorithm | 2.0 | 0.7 | 0.6 † | 1.2 | 159 |
| `3477fb2` | 2026-06-13 | modified SAH, clustering instead of morton for SOA triangles grouping | 5.1 | 1.9 | 1.6 † | 2.7 | 265 |
| `db51ac1` | 2026-06-13 | cleanup comments, and magic numbers | 5.2 | 1.9 | 1.6 † | 2.7 | 267 |
| `b1fb0ba` | 2026-06-13 | tweaks | 5.2 | 2.0 | 1.6 † | 2.8 | 284 |
| `0d2606b` | 2026-06-13 | add textures support | 5.2 | 1.9 | 1.6 † | 2.7 | 292 |
| `0778479` | 2026-06-13 | fix warnings | 5.2 | 1.9 | 1.6 † | 2.7 | 287 |
| `ec4fa65` | 2026-06-14 | optimize bvh | 5.9 | 2.4 | 1.8 † | 3.0 | 291 |
| `73eb3ff` | 2026-06-14 | slight bvh optimisation | 6.7 | 2.6 | 2.0 † | 3.2 | 289 |
| `c0e2370` | 2026-06-14 | bvh8 attempt | 10.8 | 4.8 | 3.4 † | 4.4 | 294 |
| `51ab953` | 2026-06-14 | mix performance tweaks | 10.0 | 4.7 | 3.3 † | 4.3 | 296 |
| `59e8bde` | 2026-06-14 | shadow ray optimization | 10.1 | 4.8 | 3.2 | 4.3 | 295 |
| `65c2817` | 2026-06-15 | refactorings | 9.9 | 4.6 | 3.1 | 4.3 | 296 |
| `b99763c` | 2026-06-15 | emissive lights | 10.1 | 4.7 | 3.1 | 4.3 | 296 |
| `19f08c5` | 2026-06-15 | generic light list (emissive mesh + anlytic) | 9.8 | 4.6 | 3.2 | 4.3 | 295 |
| `f0d9350` | 2026-06-16 | cleanup, add comments | 10.3 | 4.8 | 3.2 | 4.3 | 297 |
| `d73438c` | 2026-06-16 | add MIS support | 10.2 | 4.8 | 3.1 | 4.5 | 302 |
| `3ce2ba2` | 2026-06-17 | add hdr sky support | 9.8 | 4.8 | 3.2 | 4.4 | 299 |
| `18cc184` | 2026-06-17 | add benchmarks | 10.2 | 4.8 | 3.1 | 4.4 | 301 |
| `befb868` | 2026-06-18 | try octant order lut to avoid sortir children in bvh traversal | 9.9 | 4.0 | 3.4 | 4.4 | 300 |
| `4a099c0` | 2026-06-18 | extract BVH into rnpt-bvh crate with clean Embree-like interface | 9.6 | 3.9 | 3.4 | 4.4 | 182 |
| `fb5869f` | 2026-06-19 | Cook Torrance/GGX and PBR materials support. | 9.4 | 3.8 | 3.4 | 4.3 | 181 |
| `e8f107d` | 2026-06-20 | glass support | 9.1 | 3.7 | 3.2 | 4.2 | 184 |
| `d17de5d` | 2026-06-23 | use FMA for ray intersection | 9.5 | 3.9 | 3.4 | 0.0 | 184 |
| `1c9d44d` | 2026-06-24 | true restir, fix regression in bvh intersection | 9.6 | 3.9 | 3.4 | 4.2 | 184 |
| `63042ea` | 2026-06-26 | cleanup | 9.5 | 3.9 | 3.4 | 4.2 | 186 |
| `c020143` | 2026-06-27 | first nirc version | 9.5 | 3.9 | 3.4 | 4.2 | 184 |
| `29cac5b` | 2026-07-16 | cleanup before going public | 9.5 | 3.9 | 3.4 | 4.2 | 185 |
| `75e6c3a` | 2026-07-16 | remove dead code | 9.6 | 3.9 | 3.4 | 4.2 | 185 |
| `cbd3127` | 2026-10-15 | bvh: revert the octant LUT (befb868), sort children by distance again | 9.8 | 4.6 | 3.1 | 4.2 | 185 |
| `d18ccbb` | 2026-10-16 | rnpt-render: headless renderer, settings saved as JSON next to every render; NIRC timing counters | 9.8 | 4.6 | 3.1 | 4.2 | 184 |

## sponza

| commit | date | subject | primary | diffuse | shadow | ortho | build ms |
|---|---|---|---:|---:|---:|---:|---:|
| `ab81978` | 2026-06-12 | add basic bvh | 2.3 | 1.1 | 0.9 † | 1.2 | 58 |
| `2ebe9f1` | 2026-06-13 | various improvements | 2.3 | 1.2 | 1.0 † | 1.3 | 57 |
| `25fe426` | 2026-06-13 | intersection simd optimization with wide | 3.1 | 1.7 | 1.4 † | 1.7 | 60 |
| `124a279` | 2026-06-13 | change the bvh algorithm | 3.1 | 1.7 | 1.4 † | 1.6 | 44 |
| `3477fb2` | 2026-06-13 | modified SAH, clustering instead of morton for SOA triangles grouping | 7.0 | 3.8 | 2.9 † | 4.2 | 70 |
| `db51ac1` | 2026-06-13 | cleanup comments, and magic numbers | 7.4 | 4.1 | 3.3 † | 4.8 | 73 |
| `b1fb0ba` | 2026-06-13 | tweaks | 7.4 | 4.1 | 3.3 † | 4.9 | 76 |
| `0d2606b` | 2026-06-13 | add textures support | 7.4 | 4.1 | 3.3 † | 4.9 | 78 |
| `0778479` | 2026-06-13 | fix warnings | 7.4 | 4.1 | 3.3 † | 4.9 | 78 |
| `ec4fa65` | 2026-06-14 | optimize bvh | 7.6 | 4.9 | 3.6 † | 4.5 | 78 |
| `73eb3ff` | 2026-06-14 | slight bvh optimisation | 9.4 | 5.7 | 4.3 † | 5.9 | 77 |
| `c0e2370` | 2026-06-14 | bvh8 attempt | 13.2 | 8.8 | 8.3 † | 8.6 | 79 |
| `51ab953` | 2026-06-14 | mix performance tweaks | 12.3 | 8.9 | 8.1 † | 7.7 | 79 |
| `59e8bde` | 2026-06-14 | shadow ray optimization | 12.3 | 8.9 | 10.6 | 7.6 | 80 |
| `65c2817` | 2026-06-15 | refactorings | 12.3 | 8.9 | 10.6 | 7.6 | 81 |
| `b99763c` | 2026-06-15 | emissive lights | 12.4 | 9.0 | 10.6 | 7.7 | 79 |
| `19f08c5` | 2026-06-15 | generic light list (emissive mesh + anlytic) | 12.3 | 9.0 | 10.6 | 7.6 | 79 |
| `f0d9350` | 2026-06-16 | cleanup, add comments | 12.3 | 9.0 | 10.5 | 7.6 | 79 |
| `d73438c` | 2026-06-16 | add MIS support | 12.3 | 9.0 | 10.5 | 7.6 | 80 |
| `3ce2ba2` | 2026-06-17 | add hdr sky support | 12.4 | 9.0 | 10.5 | 7.7 | 80 |
| `18cc184` | 2026-06-17 | add benchmarks | 12.4 | 9.0 | 10.5 | 7.6 | 80 |
| `befb868` | 2026-06-18 | try octant order lut to avoid sortir children in bvh traversal | 10.2 | 8.2 | 7.3 | 8.6 | 82 |
| `4a099c0` | 2026-06-18 | extract BVH into rnpt-bvh crate with clean Embree-like interface | 9.7 | 7.6 | 6.9 | 8.1 | 47 |
| `fb5869f` | 2026-06-19 | Cook Torrance/GGX and PBR materials support. | 9.7 | 7.6 | 6.9 | 8.1 | 48 |
| `e8f107d` | 2026-06-20 | glass support | 9.0 | 7.2 | 6.4 | 7.4 | 48 |
| `d17de5d` | 2026-06-23 | use FMA for ray intersection | 9.6 | 7.6 | 6.8 | 0.0 ✗ | 48 |
| `1c9d44d` | 2026-06-24 | true restir, fix regression in bvh intersection | 9.5 | 7.5 | 6.7 | 7.9 | 49 |
| `63042ea` | 2026-06-26 | cleanup | 9.6 | 7.5 | 6.7 | 7.9 | 49 |
| `c020143` | 2026-06-27 | first nirc version | 9.5 | 7.6 | 6.7 | 7.9 | 49 |
| `29cac5b` | 2026-07-16 | cleanup before going public | 9.6 | 7.6 | 6.7 | 7.9 | 48 |
| `75e6c3a` | 2026-07-16 | remove dead code | 9.6 | 7.6 | 6.7 | 7.9 | 48 |
| `cbd3127` | 2026-10-15 | bvh: revert the octant LUT (befb868), sort children by distance again | 11.1 | 8.2 | 9.5 | 7.1 | 49 |
| `d18ccbb` | 2026-10-16 | rnpt-render: headless renderer, settings saved as JSON next to every render; NIRC timing counters | 11.2 | 8.2 | 9.5 | 7.1 | 48 |
| `0a05f36` | 2026-10-15 | bvh step 14: back to (p - o) * inv_dir in the box test, fixes axis-aligned rays (1c9d44d) | 9.4 | 7.4 | 6.6 | 7.8 | 49 |
| `14da82f` | 2026-10-15 | bvh step 15 (experiment): closest hit sorts children again, the octant LUT stays unused | 10.4 | 7.9 | 6.7 | 7.2 | 49 |
| `379ff20` | 2026-10-15 | bvh step 7: traversal stack without bounds checks (73eb3ff) | 8.2 | 5.0 | 3.9 | 5.3 | 50 |
| `4ea22b1` | 2026-10-15 | bvh step 1: front-to-back traversal, prune nodes behind the closest hit (2ebe9f1) | 2.3 | 1.2 | 1.0 | 1.3 | 30 |
| `797a814` | 2026-10-15 | bvh step 17: revert the octant LUT (befb868): sort children by distance again, slots in build order | 11.0 | 8.0 | 9.3 | 7.0 | 49 |
| `867fb20` | 2026-10-15 | bvh step 0: binary BVH, Morton order, median split (ab81978) | 2.3 | 1.1 | 0.9 | 1.2 | 30 |
| `96c3a79` | 2026-10-15 | bvh step 5: 16 SAH bins instead of 8 (db51ac1, b1fb0ba) | 6.9 | 3.8 | 3.1 | 4.6 | 51 |
| `ae1542e` | 2026-10-15 | bvh step 3: pack 8 consecutive triangles into chunks first, build the tree over chunks (124a279) | 3.0 | 1.6 | 1.4 | 1.6 | 16 |
| `c0ce714` | 2026-10-15 | bvh step 12: octant order LUT instead of sorting children, octant slots in the builder (befb868) | 8.9 | 7.1 | 6.3 | 7.3 | 49 |
| `c1a0715` | 2026-10-15 | bvh step 4: cluster triangles into chunks by median splits, binned SAH over chunks (3477fb2) | 6.6 | 3.6 | 2.8 | 4.1 | 44 |
| `c433e1b` | 2026-10-15 | bvh step 16 (experiment): any-hit visits children near-first using the octant LUT | 10.4 | 7.9 | 8.1 | 7.2 | 49 |
| `dee98f2` | 2026-10-15 | bvh step 11: final code layout (leaf encoding helpers, TriSlot, per-chunk double-sided masks), same algorithm (65c2817, 4a099c0, e8f107d) | 10.6 | 7.7 | 8.9 | 6.7 | 49 |
| `e27585e` | 2026-10-15 | bvh step 6: precomputed reciprocal direction, branchless slab test, unchecked node access (ec4fa65) | 6.8 | 4.4 | 3.3 | 4.2 | 51 |
| `f0126d9` | 2026-10-15 | bvh step 2: 8-triangle SoA leaves, SIMD intersection with wide (25fe426) | 3.0 | 1.6 | 1.4 | 1.6 | 29 |
