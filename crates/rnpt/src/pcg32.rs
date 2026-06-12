pub struct Pcg32 {
    state: u64,
    inc: u64, // Controls the sequence (stream ID), must always be odd
}

impl Pcg32 {
    // Standard multiplier constant defined by the PCG paper
    const MULTIPLIER: u64 = 6364136223846793005;

    /// Creates a new PCG32 generator.
    /// init_state: Initial seed
    /// init_seq: Stream ID (sequence). Will be forced to be odd.
    pub fn from_seed(init_state: u64, init_seq: u64) -> Self {
        let mut rng = Self {
            state: 0,
            // The increment must be odd to ensure the LCG has a full period
            inc: (init_seq << 1) | 1,
        };

        // Initialize the state by stepping once
        rng.state = rng.state.wrapping_add(init_state);
        rng.next_u32();
        rng
    }

    pub fn from_seed_128(seed: u128) -> Self {
        // Extract the upper and lower 64 bits from the 128-bit block
        let init_state = (seed >> 64) as u64;
        let init_seq = seed as u64;

        let mut rng = Self {
            state: 0,
            // The stream ID (lower 64 bits) is forced to be odd
            inc: (init_seq << 1) | 1,
        };

        // Warm-up cycles to thoroughly mix the bits
        rng.next_u32();
        rng.state = rng.state.wrapping_add(init_state);
        rng.next_u32();

        rng
    }

    /// Generates the next pseudo-random 32-bit unsigned integer.
    pub fn next_u32(&mut self) -> u32 {
        let old_state = self.state;

        // 1. Advance the internal Linear Congruential Generator (LCG)
        self.state = old_state
            .wrapping_mul(Self::MULTIPLIER)
            .wrapping_add(self.inc);

        // 2. Apply the XSH (XorShift High) transformation
        // Bring high-entropy bits down to the middle
        let xorshifted = (((old_state >> 18) ^ old_state) >> 27) as u32;

        // 3. Apply the RR (Random Rotate) transformation
        // Use the topmost 5 bits of the old state to determine the rotation count
        let rot = (old_state >> 59) as u32;

        // Rust compiles .rotate_right directly into a single hardware instruction (ROR)
        xorshifted.rotate_right(rot)
    }

    /// Generates a float uniformly distributed in the half-open interval [0.0, 1.0)
    pub fn next_f32(&mut self) -> f32 {
        // Divide by 2^32 to map exactly into [0.0, 1.0)
        // Multiplying by the constant reciprocal is faster than division
        const FACTOR: f32 = 1.0 / (u32::MAX as f64 + 1.0) as f32;
        (self.next_u32() as f32) * FACTOR
    }
}
