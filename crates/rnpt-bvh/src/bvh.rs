use nalgebra::Point3;

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

#[derive(Clone, Copy)]
pub struct BvhNode {
    pub aabb: AABB,
    pub left_first: u32,
    pub tri_count: u32,
}

impl BvhNode {
    pub fn is_leaf(&self) -> bool {
        self.tri_count > 0
    }
}

pub struct BvhInner {
    pub nodes: Vec<BvhNode>,
    pub vertices: Vec<Point3<f32>>,
    pub triangles: Vec<FlatTriangle>,
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
                let first = node.left_first as usize;
                let count = node.tri_count as usize;

                for i in 0..count {
                    let tri = &self.triangles[first + i];
                    let v0 = self.vertices[tri.v0 as usize];
                    let v1 = self.vertices[tri.v1 as usize];
                    let v2 = self.vertices[tri.v2 as usize];

                    let mut test_ray = ray.clone();
                    test_ray.tmax = t_max;

                    if let Some(hit) = test_ray.intersect_triangle(&v0, &v1, &v2, tri.double_sided) {
                        t_max = hit.t;
                        result = Some(Hit {
                            t: hit.t,
                            u: hit.u,
                            v: hit.v,
                            prim_id: tri.orig_idx,
                            geom_id: tri.geom_id,
                        });
                    }
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
