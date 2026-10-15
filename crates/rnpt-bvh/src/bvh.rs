use crate::math::{InternalRay, AABB};
use crate::Hit;

const MAX_TRAVERSAL_DEPTH: usize = 64;

#[derive(Clone, Copy)]
pub struct FlatTriangle {
    pub v0: u32,
    pub v1: u32,
    pub v2: u32,
    pub orig_idx: u32,
    pub geom_id: u32,
    pub double_sided: bool,
}

/// 8 consecutive triangles laid out for SIMD, one lane per triangle: chunk `c`
/// holds triangles `8c..8c+8`. Padding lanes are degenerate and never hit.
#[repr(C, align(32))]
#[derive(Clone, Copy)]
pub struct FlatTriangles {
    pub v0_x: [f32; 8], pub v0_y: [f32; 8], pub v0_z: [f32; 8],
    pub e1_x: [f32; 8], pub e1_y: [f32; 8], pub e1_z: [f32; 8],
    pub e2_x: [f32; 8], pub e2_y: [f32; 8], pub e2_z: [f32; 8],
    /// Bit i set: lane i is double-sided.
    pub double_sided_mask: u8,
}

impl Default for FlatTriangles {
    fn default() -> Self {
        Self {
            v0_x: [0.0; 8], v0_y: [0.0; 8], v0_z: [0.0; 8],
            e1_x: [0.0; 8], e1_y: [0.0; 8], e1_z: [0.0; 8],
            e2_x: [0.0; 8], e2_y: [0.0; 8], e2_z: [0.0; 8],
            double_sided_mask: 0,
        }
    }
}

#[derive(Clone, Copy)]
pub struct BvhNode {
    pub aabb: AABB,
    /// Internal node: index of the left child. Leaf: index of its first chunk.
    pub left_first: u32,
    /// Number of chunks in a leaf, 0 for an internal node.
    pub tri_count: u32,
}

impl BvhNode {
    pub fn is_leaf(&self) -> bool {
        self.tri_count > 0
    }
}

pub struct BvhInner {
    pub nodes: Vec<BvhNode>,
    pub triangles: Vec<FlatTriangle>,
    pub soa_chunks: Vec<FlatTriangles>,
}

impl BvhInner {
    pub fn closest_hit(&self, ray: &InternalRay) -> Option<Hit> {
        if self.nodes.is_empty() {
            return None;
        }

        let mut result: Option<Hit> = None;
        let mut t_max = ray.tmax;

        // Each stack entry carries the distance at which the ray enters the node,
        // so nodes behind the closest hit found so far are skipped when popped.
        let mut stack = [(0usize, 0.0f32); MAX_TRAVERSAL_DEPTH];
        let mut stack_ptr = 0;

        let Some(root_t) = ray.intersect_aabb(&self.nodes[0].aabb, t_max) else {
            return None;
        };
        stack[stack_ptr] = (0, root_t);
        stack_ptr += 1;

        while stack_ptr > 0 {
            stack_ptr -= 1;
            let (node_idx, node_t) = stack[stack_ptr];

            if node_t >= t_max {
                continue;
            }

            let node = &self.nodes[node_idx];

            if node.is_leaf() {
                let chunk_idx = node.left_first as usize;
                let chunk = &self.soa_chunks[chunk_idx];

                if let Some(simd_hit) = ray.intersect_simd_8(chunk, t_max) {
                    t_max = simd_hit.hit.t;
                    let tri = &self.triangles[chunk_idx * 8 + simd_hit.lane];
                    result = Some(Hit {
                        t: simd_hit.hit.t,
                        u: simd_hit.hit.u,
                        v: simd_hit.hit.v,
                        prim_id: tri.orig_idx,
                        geom_id: tri.geom_id,
                    });
                }
            } else if stack_ptr + 1 < MAX_TRAVERSAL_DEPTH {
                let left_idx = node.left_first as usize;
                let right_idx = left_idx + 1;
                let t_left = ray.intersect_aabb(&self.nodes[left_idx].aabb, t_max);
                let t_right = ray.intersect_aabb(&self.nodes[right_idx].aabb, t_max);

                match (t_left, t_right) {
                    (Some(t_left), Some(t_right)) => {
                        // Front to back: push the farthest first, so the nearest is popped first.
                        if t_left < t_right {
                            stack[stack_ptr] = (right_idx, t_right);
                            stack[stack_ptr + 1] = (left_idx, t_left);
                        } else {
                            stack[stack_ptr] = (left_idx, t_left);
                            stack[stack_ptr + 1] = (right_idx, t_right);
                        }
                        stack_ptr += 2;
                    }
                    (Some(t_left), None) => {
                        stack[stack_ptr] = (left_idx, t_left);
                        stack_ptr += 1;
                    }
                    (None, Some(t_right)) => {
                        stack[stack_ptr] = (right_idx, t_right);
                        stack_ptr += 1;
                    }
                    (None, None) => {}
                }
            }
        }

        result
    }

    /// No dedicated occlusion query yet: any hit is a closest hit.
    pub fn any_hit(&self, ray: &InternalRay) -> bool {
        self.closest_hit(ray).is_some()
    }
}
