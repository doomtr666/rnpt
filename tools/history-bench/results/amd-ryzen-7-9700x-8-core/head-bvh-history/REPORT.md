# BVH history — amd-ryzen-7-9700x-8-core — build config: head

Single-thread throughput in Mrays/s: median over processes of the median pass. ✗ = results disagree with the brute-force reference; † = no occlusion query yet, shadow rays use closest-hit.

## bistro-ext

| commit | date | subject | primary | diffuse | shadow | ortho | build ms |
|---|---|---|---:|---:|---:|---:|---:|
| `c623596` | 2026-10-15 | tools: history-bench, replays the BVH history on fixed inputs | 9.6 | 2.6 | 4.2 | 7.3 | 587 |
| `867fb20` | 2026-10-15 | bvh step 0: binary BVH, Morton order, median split (ab81978) | 1.5 | 0.2 | 0.2 | 0.8 | 405 |
| `4ea22b1` | 2026-10-15 | bvh step 1: front-to-back traversal, prune nodes behind the closest hit (2ebe9f1) | 1.1 | 0.3 | 0.3 | 0.7 | 408 |
| `f0126d9` | 2026-10-15 | bvh step 2: 8-triangle SoA leaves, SIMD intersection with wide (25fe426) | 1.4 | 0.4 | 0.4 | 0.9 | 408 |
| `ae1542e` | 2026-10-15 | bvh step 3: pack 8 consecutive triangles into chunks first, build the tree over chunks (124a279) | 1.5 | 0.4 | 0.4 | 1.0 | 206 |
| `c1a0715` | 2026-10-15 | bvh step 4: cluster triangles into chunks by median splits, binned SAH over chunks (3477fb2) | 5.3 | 1.4 | 1.4 | 4.5 | 524 |
| `96c3a79` | 2026-10-15 | bvh step 5: 16 SAH bins instead of 8 (db51ac1, b1fb0ba) | 5.5 | 1.4 | 1.5 | 4.4 | 571 |
| `e27585e` | 2026-10-15 | bvh step 6: precomputed reciprocal direction, branchless slab test, unchecked node access (ec4fa65) | 5.9 | 1.5 | 1.6 | 4.5 | 572 |
| `379ff20` | 2026-10-15 | bvh step 7: traversal stack without bounds checks (73eb3ff) | 6.7 | 1.7 | 1.8 | 5.1 | 566 |
| `e79f25b` | 2026-10-15 | bvh step 8: collapse the binary tree into an 8-wide BVH, 8 boxes per SIMD test (c0e2370) | 10.1 | 2.9 | 3.1 | 7.3 | 576 |
| `424a87e` | 2026-10-15 | bvh step 9: pre-encoded children, uninitialized stack, unchecked chunk access, branchless lane min (51ab953) | 10.1 | 2.9 | 3.1 | 7.1 | 575 |
| `509a435` | 2026-10-15 | bvh step 10: dedicated any-hit query for shadow rays (59e8bde) | 9.9 | 2.9 | 4.3 | 7.0 | 598 |
| `dee98f2` | 2026-10-15 | bvh step 11: final code layout (leaf encoding helpers, TriSlot, per-chunk double-sided masks), same algorithm (65c2817, 4a099c0, e8f107d) | 10.1 | 2.9 | 4.3 | 7.1 | 575 |
| `c0ce714` | 2026-10-15 | bvh step 12: octant order LUT instead of sorting children, octant slots in the builder (befb868) | 9.1 | 2.5 | 3.9 | 6.9 | 560 |
| `c46ee6a` | 2026-10-15 | bvh step 13: FMA in the triangle and box tests (d17de5d) | 9.1 | 2.6 | 4.1 | 0.0 | 562 |
| `0a05f36` | 2026-10-15 | bvh step 14: back to (p - o) * inv_dir in the box test, fixes axis-aligned rays (1c9d44d) | 9.5 | 2.6 | 4.2 | 7.2 | 561 |
| `14da82f` | 2026-10-15 | bvh step 15 (experiment): closest hit sorts children again, the octant LUT stays unused | 10.2 | 2.8 | 4.1 | 7.4 | 561 |
| `c433e1b` | 2026-10-15 | bvh step 16 (experiment): any-hit visits children near-first using the octant LUT | 10.5 | 3.0 | 3.8 | 7.5 | 564 |
| `797a814` | 2026-10-15 | bvh step 17: revert the octant LUT (befb868): sort children by distance again, slots in build order | 10.5 | 3.0 | 4.5 | 7.5 | 572 |

## cluster-100k

| commit | date | subject | primary | diffuse | shadow | ortho | build ms |
|---|---|---|---:|---:|---:|---:|---:|
| `c623596` | 2026-10-15 | tools: history-bench, replays the BVH history on fixed inputs | 15.5 | 5.7 | 5.7 | 13.8 | 17 |
| `867fb20` | 2026-10-15 | bvh step 0: binary BVH, Morton order, median split (ab81978) | 3.0 | 0.9 | 0.7 | 2.9 | 11 |
| `4ea22b1` | 2026-10-15 | bvh step 1: front-to-back traversal, prune nodes behind the closest hit (2ebe9f1) | 2.8 | 1.0 | 0.8 | 2.8 | 10 |
| `f0126d9` | 2026-10-15 | bvh step 2: 8-triangle SoA leaves, SIMD intersection with wide (25fe426) | 3.5 | 1.2 | 1.1 | 3.6 | 10 |
| `ae1542e` | 2026-10-15 | bvh step 3: pack 8 consecutive triangles into chunks first, build the tree over chunks (124a279) | 3.7 | 1.3 | 1.1 | 3.8 | 5 |
| `c1a0715` | 2026-10-15 | bvh step 4: cluster triangles into chunks by median splits, binned SAH over chunks (3477fb2) | 8.0 | 2.7 | 2.5 | 8.4 | 15 |
| `96c3a79` | 2026-10-15 | bvh step 5: 16 SAH bins instead of 8 (db51ac1, b1fb0ba) | 8.2 | 2.7 | 2.5 | 8.6 | 17 |
| `e27585e` | 2026-10-15 | bvh step 6: precomputed reciprocal direction, branchless slab test, unchecked node access (ec4fa65) | 8.9 | 3.5 | 3.0 | 8.6 | 17 |
| `379ff20` | 2026-10-15 | bvh step 7: traversal stack without bounds checks (73eb3ff) | 10.0 | 3.8 | 3.3 | 9.8 | 16 |
| `e79f25b` | 2026-10-15 | bvh step 8: collapse the binary tree into an 8-wide BVH, 8 boxes per SIMD test (c0e2370) | 16.2 | 6.5 | 5.4 | 14.2 | 17 |
| `424a87e` | 2026-10-15 | bvh step 9: pre-encoded children, uninitialized stack, unchecked chunk access, branchless lane min (51ab953) | 15.9 | 6.5 | 5.4 | 13.2 | 17 |
| `509a435` | 2026-10-15 | bvh step 10: dedicated any-hit query for shadow rays (59e8bde) | 15.9 | 6.6 | 6.9 | 13.2 | 18 |
| `dee98f2` | 2026-10-15 | bvh step 11: final code layout (leaf encoding helpers, TriSlot, per-chunk double-sided masks), same algorithm (65c2817, 4a099c0, e8f107d) | 15.8 | 6.6 | 6.8 | 13.2 | 17 |
| `c0ce714` | 2026-10-15 | bvh step 12: octant order LUT instead of sorting children, octant slots in the builder (befb868) | 15.1 | 5.6 | 5.5 | 13.6 | 17 |
| `c46ee6a` | 2026-10-15 | bvh step 13: FMA in the triangle and box tests (d17de5d) | 15.4 | 5.7 | 5.6 | 0.0 ✗ | 18 |
| `0a05f36` | 2026-10-15 | bvh step 14: back to (p - o) * inv_dir in the box test, fixes axis-aligned rays (1c9d44d) | 15.4 | 5.7 | 5.7 | 13.8 | 17 |
| `14da82f` | 2026-10-15 | bvh step 15 (experiment): closest hit sorts children again, the octant LUT stays unused | 16.1 | 6.4 | 5.8 | 13.3 | 17 |
| `c433e1b` | 2026-10-15 | bvh step 16 (experiment): any-hit visits children near-first using the octant LUT | 16.2 | 6.4 | 5.7 | 13.3 | 17 |
| `797a814` | 2026-10-15 | bvh step 17: revert the octant LUT (befb868): sort children by distance again, slots in build order | 16.0 | 6.7 | 7.0 | 13.4 | 17 |

## cluster-1m

| commit | date | subject | primary | diffuse | shadow | ortho | build ms |
|---|---|---|---:|---:|---:|---:|---:|
| `c623596` | 2026-10-15 | tools: history-bench, replays the BVH history on fixed inputs | 6.9 | 4.2 | 3.8 | 5.4 | 188 |
| `867fb20` | 2026-10-15 | bvh step 0: binary BVH, Morton order, median split (ab81978) | 1.4 | 0.5 | 0.3 | 1.3 | 119 |
| `4ea22b1` | 2026-10-15 | bvh step 1: front-to-back traversal, prune nodes behind the closest hit (2ebe9f1) | 1.0 | 0.6 | 0.4 | 1.0 | 119 |
| `f0126d9` | 2026-10-15 | bvh step 2: 8-triangle SoA leaves, SIMD intersection with wide (25fe426) | 1.4 | 0.8 | 0.6 | 1.5 | 113 |
| `ae1542e` | 2026-10-15 | bvh step 3: pack 8 consecutive triangles into chunks first, build the tree over chunks (124a279) | 1.4 | 0.8 | 0.6 | 1.6 | 59 |
| `c1a0715` | 2026-10-15 | bvh step 4: cluster triangles into chunks by median splits, binned SAH over chunks (3477fb2) | 3.8 | 2.0 | 1.8 | 3.8 | 165 |
| `96c3a79` | 2026-10-15 | bvh step 5: 16 SAH bins instead of 8 (db51ac1, b1fb0ba) | 3.8 | 2.0 | 1.8 | 3.9 | 186 |
| `e27585e` | 2026-10-15 | bvh step 6: precomputed reciprocal direction, branchless slab test, unchecked node access (ec4fa65) | 4.1 | 2.5 | 2.0 | 3.8 | 188 |
| `379ff20` | 2026-10-15 | bvh step 7: traversal stack without bounds checks (73eb3ff) | 4.7 | 2.7 | 2.3 | 4.4 | 187 |
| `e79f25b` | 2026-10-15 | bvh step 8: collapse the binary tree into an 8-wide BVH, 8 boxes per SIMD test (c0e2370) | 7.3 | 4.8 | 3.8 | 6.3 | 192 |
| `424a87e` | 2026-10-15 | bvh step 9: pre-encoded children, uninitialized stack, unchecked chunk access, branchless lane min (51ab953) | 7.0 | 4.8 | 3.7 | 5.8 | 192 |
| `509a435` | 2026-10-15 | bvh step 10: dedicated any-hit query for shadow rays (59e8bde) | 7.0 | 4.8 | 4.1 | 5.9 | 193 |
| `dee98f2` | 2026-10-15 | bvh step 11: final code layout (leaf encoding helpers, TriSlot, per-chunk double-sided masks), same algorithm (65c2817, 4a099c0, e8f107d) | 7.0 | 4.8 | 4.1 | 5.8 | 188 |
| `c0ce714` | 2026-10-15 | bvh step 12: octant order LUT instead of sorting children, octant slots in the builder (befb868) | 6.8 | 4.2 | 3.7 | 5.4 | 187 |
| `c46ee6a` | 2026-10-15 | bvh step 13: FMA in the triangle and box tests (d17de5d) | 6.8 | 4.2 | 3.7 | 0.0 ✗ | 190 |
| `0a05f36` | 2026-10-15 | bvh step 14: back to (p - o) * inv_dir in the box test, fixes axis-aligned rays (1c9d44d) | 6.8 | 4.2 | 3.8 | 5.4 | 188 |
| `14da82f` | 2026-10-15 | bvh step 15 (experiment): closest hit sorts children again, the octant LUT stays unused | 7.3 | 4.7 | 3.9 | 6.1 | 188 |
| `c433e1b` | 2026-10-15 | bvh step 16 (experiment): any-hit visits children near-first using the octant LUT | 7.3 | 4.7 | 4.1 | 6.1 | 188 |
| `797a814` | 2026-10-15 | bvh step 17: revert the octant LUT (befb868): sort children by distance again, slots in build order | 7.1 | 4.9 | 4.2 | 6.0 | 187 |

## hf-100k

| commit | date | subject | primary | diffuse | shadow | ortho | build ms |
|---|---|---|---:|---:|---:|---:|---:|
| `c623596` | 2026-10-15 | tools: history-bench, replays the BVH history on fixed inputs | 9.9 | 5.1 | 5.1 | 7.2 | 17 |
| `867fb20` | 2026-10-15 | bvh step 0: binary BVH, Morton order, median split (ab81978) | 2.3 | 0.8 | 0.5 | 1.5 | 10 |
| `4ea22b1` | 2026-10-15 | bvh step 1: front-to-back traversal, prune nodes behind the closest hit (2ebe9f1) | 2.2 | 0.9 | 0.7 | 1.5 | 10 |
| `f0126d9` | 2026-10-15 | bvh step 2: 8-triangle SoA leaves, SIMD intersection with wide (25fe426) | 3.0 | 1.1 | 1.0 | 2.1 | 10 |
| `ae1542e` | 2026-10-15 | bvh step 3: pack 8 consecutive triangles into chunks first, build the tree over chunks (124a279) | 3.2 | 1.2 | 1.1 | 2.2 | 5 |
| `c1a0715` | 2026-10-15 | bvh step 4: cluster triangles into chunks by median splits, binned SAH over chunks (3477fb2) | 6.5 | 2.5 | 2.4 | 4.8 | 15 |
| `96c3a79` | 2026-10-15 | bvh step 5: 16 SAH bins instead of 8 (db51ac1, b1fb0ba) | 6.6 | 2.6 | 2.4 | 4.9 | 17 |
| `e27585e` | 2026-10-15 | bvh step 6: precomputed reciprocal direction, branchless slab test, unchecked node access (ec4fa65) | 7.2 | 3.1 | 2.7 | 4.9 | 17 |
| `379ff20` | 2026-10-15 | bvh step 7: traversal stack without bounds checks (73eb3ff) | 8.1 | 3.4 | 3.0 | 5.4 | 17 |
| `e79f25b` | 2026-10-15 | bvh step 8: collapse the binary tree into an 8-wide BVH, 8 boxes per SIMD test (c0e2370) | 12.3 | 5.9 | 4.9 | 7.9 | 18 |
| `424a87e` | 2026-10-15 | bvh step 9: pre-encoded children, uninitialized stack, unchecked chunk access, branchless lane min (51ab953) | 11.7 | 6.0 | 4.8 | 7.1 | 17 |
| `509a435` | 2026-10-15 | bvh step 10: dedicated any-hit query for shadow rays (59e8bde) | 11.8 | 6.0 | 4.9 | 7.1 | 18 |
| `dee98f2` | 2026-10-15 | bvh step 11: final code layout (leaf encoding helpers, TriSlot, per-chunk double-sided masks), same algorithm (65c2817, 4a099c0, e8f107d) | 11.7 | 6.0 | 4.9 | 7.1 | 17 |
| `c0ce714` | 2026-10-15 | bvh step 12: octant order LUT instead of sorting children, octant slots in the builder (befb868) | 9.7 | 5.0 | 5.0 | 7.1 | 17 |
| `c46ee6a` | 2026-10-15 | bvh step 13: FMA in the triangle and box tests (d17de5d) | 9.9 | 5.2 | 5.0 | 0.0 | 17 |
| `0a05f36` | 2026-10-15 | bvh step 14: back to (p - o) * inv_dir in the box test, fixes axis-aligned rays (1c9d44d) | 10.0 | 5.1 | 5.1 | 7.2 | 17 |
| `14da82f` | 2026-10-15 | bvh step 15 (experiment): closest hit sorts children again, the octant LUT stays unused | 11.8 | 5.8 | 5.2 | 7.0 | 17 |
| `c433e1b` | 2026-10-15 | bvh step 16 (experiment): any-hit visits children near-first using the octant LUT | 11.8 | 5.8 | 4.9 | 7.0 | 17 |
| `797a814` | 2026-10-15 | bvh step 17: revert the octant LUT (befb868): sort children by distance again, slots in build order | 11.9 | 6.2 | 5.0 | 7.2 | 17 |

## hf-1m

| commit | date | subject | primary | diffuse | shadow | ortho | build ms |
|---|---|---|---:|---:|---:|---:|---:|
| `c623596` | 2026-10-15 | tools: history-bench, replays the BVH history on fixed inputs | 9.6 | 3.9 | 3.4 | 4.2 | 184 |
| `867fb20` | 2026-10-15 | bvh step 0: binary BVH, Morton order, median split (ab81978) | 1.6 | 0.5 | 0.3 | 0.9 | 116 |
| `4ea22b1` | 2026-10-15 | bvh step 1: front-to-back traversal, prune nodes behind the closest hit (2ebe9f1) | 1.5 | 0.6 | 0.4 | 0.9 | 116 |
| `f0126d9` | 2026-10-15 | bvh step 2: 8-triangle SoA leaves, SIMD intersection with wide (25fe426) | 1.9 | 0.7 | 0.6 | 1.2 | 109 |
| `ae1542e` | 2026-10-15 | bvh step 3: pack 8 consecutive triangles into chunks first, build the tree over chunks (124a279) | 1.9 | 0.7 | 0.6 | 1.2 | 55 |
| `c1a0715` | 2026-10-15 | bvh step 4: cluster triangles into chunks by median splits, binned SAH over chunks (3477fb2) | 4.9 | 1.8 | 1.6 | 2.6 | 160 |
| `96c3a79` | 2026-10-15 | bvh step 5: 16 SAH bins instead of 8 (db51ac1, b1fb0ba) | 5.0 | 1.9 | 1.5 | 2.6 | 183 |
| `e27585e` | 2026-10-15 | bvh step 6: precomputed reciprocal direction, branchless slab test, unchecked node access (ec4fa65) | 5.6 | 2.3 | 1.8 | 2.8 | 184 |
| `379ff20` | 2026-10-15 | bvh step 7: traversal stack without bounds checks (73eb3ff) | 6.2 | 2.5 | 1.9 | 3.0 | 183 |
| `e79f25b` | 2026-10-15 | bvh step 8: collapse the binary tree into an 8-wide BVH, 8 boxes per SIMD test (c0e2370) | 10.1 | 4.5 | 3.3 | 3.9 | 189 |
| `424a87e` | 2026-10-15 | bvh step 9: pre-encoded children, uninitialized stack, unchecked chunk access, branchless lane min (51ab953) | 9.5 | 4.5 | 3.2 | 4.0 | 188 |
| `509a435` | 2026-10-15 | bvh step 10: dedicated any-hit query for shadow rays (59e8bde) | 8.9 | 4.3 | 2.9 | 3.9 | 192 |
| `dee98f2` | 2026-10-15 | bvh step 11: final code layout (leaf encoding helpers, TriSlot, per-chunk double-sided masks), same algorithm (65c2817, 4a099c0, e8f107d) | 9.5 | 4.4 | 3.0 | 4.1 | 184 |
| `c0ce714` | 2026-10-15 | bvh step 12: octant order LUT instead of sorting children, octant slots in the builder (befb868) | 9.5 | 3.8 | 3.3 | 4.2 | 183 |
| `c46ee6a` | 2026-10-15 | bvh step 13: FMA in the triangle and box tests (d17de5d) | 8.9 | 3.8 | 3.3 | 0.0 | 186 |
| `0a05f36` | 2026-10-15 | bvh step 14: back to (p - o) * inv_dir in the box test, fixes axis-aligned rays (1c9d44d) | 8.9 | 3.7 | 3.3 | 4.1 | 184 |
| `14da82f` | 2026-10-15 | bvh step 15 (experiment): closest hit sorts children again, the octant LUT stays unused | 9.5 | 4.3 | 3.4 | 4.2 | 184 |
| `c433e1b` | 2026-10-15 | bvh step 16 (experiment): any-hit visits children near-first using the octant LUT | 9.8 | 4.5 | 3.1 | 4.3 | 183 |
| `797a814` | 2026-10-15 | bvh step 17: revert the octant LUT (befb868): sort children by distance again, slots in build order | 10.1 | 4.7 | 3.2 | 4.3 | 183 |

## sponza

| commit | date | subject | primary | diffuse | shadow | ortho | build ms |
|---|---|---|---:|---:|---:|---:|---:|
| `c623596` | 2026-10-15 | tools: history-bench, replays the BVH history on fixed inputs | 9.5 | 7.5 | 6.7 | 7.9 | 48 |
| `867fb20` | 2026-10-15 | bvh step 0: binary BVH, Morton order, median split (ab81978) | 2.4 | 1.1 | 0.9 | 1.2 | 29 |
| `4ea22b1` | 2026-10-15 | bvh step 1: front-to-back traversal, prune nodes behind the closest hit (2ebe9f1) | 2.3 | 1.2 | 1.0 | 1.3 | 29 |
| `f0126d9` | 2026-10-15 | bvh step 2: 8-triangle SoA leaves, SIMD intersection with wide (25fe426) | 3.0 | 1.7 | 1.4 | 1.6 | 28 |
| `ae1542e` | 2026-10-15 | bvh step 3: pack 8 consecutive triangles into chunks first, build the tree over chunks (124a279) | 3.0 | 1.6 | 1.4 | 1.6 | 16 |
| `c1a0715` | 2026-10-15 | bvh step 4: cluster triangles into chunks by median splits, binned SAH over chunks (3477fb2) | 6.7 | 3.7 | 2.8 | 4.2 | 43 |
| `96c3a79` | 2026-10-15 | bvh step 5: 16 SAH bins instead of 8 (db51ac1, b1fb0ba) | 7.1 | 3.9 | 3.2 | 4.7 | 48 |
| `e27585e` | 2026-10-15 | bvh step 6: precomputed reciprocal direction, branchless slab test, unchecked node access (ec4fa65) | 7.0 | 4.5 | 3.4 | 4.3 | 48 |
| `379ff20` | 2026-10-15 | bvh step 7: traversal stack without bounds checks (73eb3ff) | 8.4 | 5.1 | 3.9 | 5.4 | 48 |
| `e79f25b` | 2026-10-15 | bvh step 8: collapse the binary tree into an 8-wide BVH, 8 boxes per SIMD test (c0e2370) | 11.1 | 7.7 | 7.3 | 7.1 | 51 |
| `424a87e` | 2026-10-15 | bvh step 9: pre-encoded children, uninitialized stack, unchecked chunk access, branchless lane min (51ab953) | 10.8 | 7.8 | 7.2 | 6.8 | 50 |
| `509a435` | 2026-10-15 | bvh step 10: dedicated any-hit query for shadow rays (59e8bde) | 10.7 | 7.8 | 9.1 | 6.8 | 50 |
| `dee98f2` | 2026-10-15 | bvh step 11: final code layout (leaf encoding helpers, TriSlot, per-chunk double-sided masks), same algorithm (65c2817, 4a099c0, e8f107d) | 10.7 | 7.7 | 9.0 | 6.8 | 48 |
| `c0ce714` | 2026-10-15 | bvh step 12: octant order LUT instead of sorting children, octant slots in the builder (befb868) | 9.1 | 7.2 | 6.4 | 7.4 | 48 |
| `c46ee6a` | 2026-10-15 | bvh step 13: FMA in the triangle and box tests (d17de5d) | 9.6 | 7.6 | 6.6 | 0.0 ✗ | 48 |
| `0a05f36` | 2026-10-15 | bvh step 14: back to (p - o) * inv_dir in the box test, fixes axis-aligned rays (1c9d44d) | 9.5 | 7.5 | 6.7 | 7.9 | 48 |
| `14da82f` | 2026-10-15 | bvh step 15 (experiment): closest hit sorts children again, the octant LUT stays unused | 10.4 | 8.0 | 6.8 | 7.3 | 48 |
| `c433e1b` | 2026-10-15 | bvh step 16 (experiment): any-hit visits children near-first using the octant LUT | 10.5 | 8.1 | 8.2 | 7.3 | 48 |
| `797a814` | 2026-10-15 | bvh step 17: revert the octant LUT (befb868): sort children by distance again, slots in build order | 11.2 | 8.2 | 9.5 | 7.1 | 48 |
