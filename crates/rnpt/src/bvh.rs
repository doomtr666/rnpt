use std::mem::MaybeUninit;

use crate::{Color, Ray, TriangleHit};
use nalgebra::{Point3, UnitVector3, Vector2};

const MAX_TRAVERSAL_DEPTH: usize = 64;

#[derive(Clone, Copy)]
pub struct FlatTriangle {
    pub v0: u32,
    pub v1: u32,
    pub v2: u32,
    pub material: u32,
}

#[repr(C, align(32))]
#[derive(Clone, Copy)]
pub struct FlatTriangles {
    pub v0_x: [f32; 8],
    pub v0_y: [f32; 8],
    pub v0_z: [f32; 8],
    pub e1_x: [f32; 8],
    pub e1_y: [f32; 8],
    pub e1_z: [f32; 8],
    pub e2_x: [f32; 8],
    pub e2_y: [f32; 8],
    pub e2_z: [f32; 8],
}

impl Default for FlatTriangles {
    fn default() -> Self {
        Self {
            v0_x: [0.0; 8],
            v0_y: [0.0; 8],
            v0_z: [0.0; 8],
            e1_x: [0.0; 8],
            e1_y: [0.0; 8],
            e1_z: [0.0; 8],
            e2_x: [0.0; 8],
            e2_y: [0.0; 8],
            e2_z: [0.0; 8],
        }
    }
}

#[repr(C, align(32))]
#[derive(Clone, Copy)]
pub struct Bvh8Node {
    pub p_min_x: [f32; 8],
    pub p_min_y: [f32; 8],
    pub p_min_z: [f32; 8],
    pub p_max_x: [f32; 8],
    pub p_max_y: [f32; 8],
    pub p_max_z: [f32; 8],
    // Pre-encoded child references (computed once at build): for a leaf,
    // `idx | (chunk_count << 24) | 0x8000_0000`; for an internal node, the
    // plain Bvh8 node index. `u32::MAX` marks an empty lane.
    pub children: [u32; 8],
}

impl Default for Bvh8Node {
    fn default() -> Self {
        Self {
            p_min_x: [f32::NAN; 8],
            p_min_y: [f32::NAN; 8],
            p_min_z: [f32::NAN; 8],
            p_max_x: [f32::NAN; 8],
            p_max_y: [f32::NAN; 8],
            p_max_z: [f32::NAN; 8],
            children: [u32::MAX; 8],
        }
    }
}

impl Bvh8Node {
    // Child reference encoding: leaves set the top bit, store the chunk count
    // in bits [24,30] and the chunk-range start in bits [0,23]. Internal nodes
    // store a plain Bvh8 node index.
    const LEAF_FLAG: u32 = 0x8000_0000;
    const COUNT_SHIFT: u32 = 24;
    const COUNT_MASK: u32 = 0x7F;
    const START_MASK: u32 = 0x00FF_FFFF;

    #[inline(always)]
    pub(crate) fn encode_leaf(start_chunk: u32, chunk_count: u32) -> u32 {
        start_chunk | (chunk_count << Self::COUNT_SHIFT) | Self::LEAF_FLAG
    }

    #[inline(always)]
    fn is_leaf(encoded: u32) -> bool {
        encoded & Self::LEAF_FLAG != 0
    }

    #[inline(always)]
    fn leaf_start(encoded: u32) -> u32 {
        encoded & Self::START_MASK
    }

    #[inline(always)]
    fn leaf_count(encoded: u32) -> u32 {
        (encoded >> Self::COUNT_SHIFT) & Self::COUNT_MASK
    }
}

pub struct BvhHit {
    pub hit: TriangleHit,
    pub material: u32,
    pub v0: u32,
    pub v1: u32,
    pub v2: u32,
    pub chunk_idx: u32,
}

pub struct Bvh {
    pub nodes: Vec<Bvh8Node>,
    pub vertices: Vec<Point3<f32>>,
    pub normals: Vec<UnitVector3<f32>>,
    pub uvs: Vec<Vector2<f32>>,
    pub colors: Vec<Color>,
    pub triangles: Vec<FlatTriangle>,
    pub soa_chunks: Vec<FlatTriangles>,
}

struct BvhStack {
    // Uninitialized on purpose: every slot is written by `push` before it can
    // be read by `pop`, so zero-initializing it per ray was a dead 512-byte
    // memset (one per Bvh::intersect call).
    data: [MaybeUninit<(u32, f32)>; MAX_TRAVERSAL_DEPTH],
    ptr: usize,
}

impl BvhStack {
    #[inline(always)]
    fn new() -> Self {
        Self {
            data: [const { MaybeUninit::uninit() }; MAX_TRAVERSAL_DEPTH],
            ptr: 0,
        }
    }

    #[inline(always)]
    fn push(&mut self, node_idx: u32, t: f32) {
        unsafe {
            self.data.get_unchecked_mut(self.ptr).write((node_idx, t));
        }
        self.ptr += 1;
    }

    #[inline(always)]
    fn pop(&mut self) -> (u32, f32) {
        self.ptr -= 1;
        unsafe { self.data.get_unchecked(self.ptr).assume_init() }
    }

    #[inline(always)]
    fn is_empty(&self) -> bool {
        self.ptr == 0
    }
}

impl Bvh {
    // Unchecked accessors for the traversal hot path. Indices are always valid
    // by construction (encoded at build time / derived from in-range lanes), so
    // we skip bounds checks while keeping `intersect` readable.
    #[inline(always)]
    fn get_node(&self, idx: u32) -> &Bvh8Node {
        unsafe { self.nodes.get_unchecked(idx as usize) }
    }

    #[inline(always)]
    fn get_chunk(&self, idx: u32) -> &FlatTriangles {
        unsafe { self.soa_chunks.get_unchecked(idx as usize) }
    }

    #[inline(always)]
    fn get_triangle(&self, idx: u32) -> &FlatTriangle {
        unsafe { self.triangles.get_unchecked(idx as usize) }
    }

    pub fn intersect(&self, ray: &Ray) -> Option<BvhHit> {
        let mut closest_hit: Option<BvhHit> = None;
        let mut t_max = ray.tmax;

        if self.nodes.is_empty() {
            return None;
        }

        let mut stack = BvhStack::new();
        stack.push(0, 0.0); // push root

        while !stack.is_empty() {
            let (encoded_idx, node_t) = stack.pop();

            if node_t >= t_max {
                continue;
            }

            if Bvh8Node::is_leaf(encoded_idx) {
                // Leaf node
                let start_chunk = Bvh8Node::leaf_start(encoded_idx);
                let chunk_count = Bvh8Node::leaf_count(encoded_idx);

                for i in 0..chunk_count {
                    let chunk_idx = start_chunk + i;
                    let chunk = self.get_chunk(chunk_idx);

                    if let Some(simd_hit) = ray.closest_triangle_simd8(chunk, t_max) {
                        t_max = simd_hit.hit.t;
                        let tri_global_idx = chunk_idx * 8 + simd_hit.lane;
                        let tri = self.get_triangle(tri_global_idx);

                        closest_hit = Some(BvhHit {
                            hit: simd_hit.hit,
                            material: tri.material,
                            v0: tri.v0,
                            v1: tri.v1,
                            v2: tri.v2,
                            chunk_idx,
                        });
                    }
                }
            } else {
                // Internal Bvh8Node
                let node = self.get_node(encoded_idx);
                let (mut bitmask, t_arr) = ray.intersect_bvh8(node, t_max);

                let mut hits = [(0u32, 0.0f32); 8];
                let mut hit_count = 0;

                while bitmask != 0 {
                    let lane = bitmask.trailing_zeros() as usize;
                    bitmask &= bitmask - 1;

                    // Encoded once at build time. Guard against empty lanes
                    // (u32::MAX) in case a NaN slips through min/max in the mask.
                    let encoded = node.children[lane];
                    let t = t_arr[lane];
                    if encoded != u32::MAX && t < t_max {
                        hits[hit_count] = (encoded, t);
                        hit_count += 1;
                    }
                }

                if hit_count > 0 {
                    // Insertion sort (descending) so the furthest is pushed first
                    for i in 1..hit_count {
                        let mut j = i;
                        while j > 0 && hits[j - 1].1 < hits[j].1 {
                            hits.swap(j - 1, j);
                            j -= 1;
                        }
                    }

                    for i in 0..hit_count {
                        stack.push(hits[i].0, hits[i].1);
                    }
                }
            }
        }

        closest_hit
    }

    /// Any-hit traversal for shadow rays: returns true as soon as any triangle
    /// is hit within the ray's range. Cheaper than `intersect`: no closest
    /// tracking, no `t_max` tightening, no child sorting, and it bails out at
    /// the first hit. `t_max` is fixed to the ray's end (e.g. light distance).
    pub fn is_occluded(&self, ray: &Ray) -> bool {
        if self.nodes.is_empty() {
            return false;
        }

        let t_max = ray.tmax;
        let mut stack = BvhStack::new();
        stack.push(0, 0.0);

        while !stack.is_empty() {
            let (encoded_idx, _) = stack.pop();

            if Bvh8Node::is_leaf(encoded_idx) {
                // Leaf node
                let start_chunk = Bvh8Node::leaf_start(encoded_idx);
                let chunk_count = Bvh8Node::leaf_count(encoded_idx);

                for i in 0..chunk_count {
                    let chunk_idx = start_chunk + i;
                    if ray.any_triangle_simd8(self.get_chunk(chunk_idx), t_max) {
                        return true; // early-out: occluder found
                    }
                }
            } else {
                // Internal node: push hit children unsorted (order is irrelevant
                // for any-hit, and t_max never tightens).
                let node = self.get_node(encoded_idx);
                let (mut bitmask, t_arr) = ray.intersect_bvh8(node, t_max);

                while bitmask != 0 {
                    let lane = bitmask.trailing_zeros() as usize;
                    bitmask &= bitmask - 1;

                    let encoded = node.children[lane];
                    if encoded != u32::MAX && t_arr[lane] < t_max {
                        stack.push(encoded, 0.0);
                    }
                }
            }
        }

        false
    }
}
