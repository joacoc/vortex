//! Split block Bloom filters implementation for vortex.
//!
//! [Split block Bloom filters]: https://arxiv.org/pdf/2101.01719

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;
use std::hash::Hasher;

use twox_hash::XxHash3_64;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

const BLOCK_SIZE: usize = 8 * size_of::<u32>();

pub struct BloomPartial {
    pub(super) blocks: Vec<[u32; 8]>,
}

impl BloomPartial {
    #[inline]
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    #[inline]
    pub(super) fn insert<T>(&mut self, value: T)
    where
        T: Hash,
    {
        let hash = self.hash(value);
        self.insert_hash(hash);
    }

    #[inline]
    pub(super) fn insert_hash(&mut self, hash: u64) {
        self.add_hash(hash);
    }

    #[inline]
    pub(super) fn hash<T>(&self, value: T) -> u64
    where
        T: Hash,
    {
        // > Since the seed is optional, it can be 0.
        // Ref: https://github.com/Cyan4973/xxHash/blob/v0.8.3/doc/xxhash_spec.md#step-1-initialize-internal-accumulators
        let mut hasher = XxHash3_64::with_seed(0);
        value.hash(&mut hasher);
        hasher.finish()
    }

    /// Hash should be u64 or u32?
    fn add_hash(&mut self, hash: u64) {
        let idx = self.block_index(hash, self.blocks.len()) as usize;
        // Block idx already consumed the hash,
        let mask = self.make_mask(hash as u32);
        for i in 0..8 {
            self.blocks[idx][i] |= mask[i];
        }
    }

    /// Checks whether a hash is (probably) present in the filter.
    pub(super) fn find_hash(&self, hash: u64) -> bool {
        let idx = self.block_index(hash, self.blocks.len()) as usize;
        let mask = self.make_mask(hash as u32);

        for i in 0..8 {
            if self.blocks[idx][i] & mask[i] != mask[i] {
                return false;
            }
        }

        true
    }

    /// Takes a hash value and creates a mask with one bit set in each 32-bit lane.
    /// These are the bits to set or check when accessing the block.
    ///
    /// The following code is SIMD friendly and will get vectorized
    /// by the compiler automatically (for releases).
    fn make_mask(&self, hash: u32) -> [u32; 8] {
        let mut out = [0u32; 8];

        // Set eight odd constants for multiply-shift hashing
        let rehash: [u32; 8] = [
            0x47b6137b, 0x44974d91, 0x8824ad5b, 0xa2b7289d, 0x705495c7, 0x2df1424b, 0x9efc4947,
            0x5c6bfb31,
        ];

        for i in 0..8 {
            // Shift all data right, reducing the hash values from 32 bits to five bits.
            // Those five bits represent an index in [0, 31)
            let y = hash.wrapping_mul(rehash[i]) >> 27;

            // Set a bit in each lane based on using the [0, 32) data as shift values.
            out[i] = 1u32 << y;
        }

        out
    }

    #[inline]
    fn block_index(&self, hash: u64, blocks_count: usize) -> u64 {
        ((hash >> 32) * (blocks_count as u64)) >> 32
    }
}

#[cfg(test)]
impl From<Vec<[u32; 8]>> for BloomPartial {
    fn from(value: Vec<[u32; 8]>) -> Self {
        BloomPartial { blocks: value }
    }
}

impl TryFrom<&[u8]> for BloomPartial {
    type Error = vortex_error::VortexError;

    /// Reconstruct a partial from its serialized byte representation
    /// (the same layout produced by `to_scalar`).
    fn try_from(bytes: &[u8]) -> VortexResult<Self> {
        vortex_ensure!(
            !bytes.is_empty() && bytes.len() % BLOCK_SIZE == 0,
            "invalid bloom filter byte length: {}",
            bytes.len()
        );

        let blocks = bytes
            .chunks_exact(BLOCK_SIZE)
            .map(|chunk| {
                let mut block = [0u32; 8];
                for (word, wb) in block.iter_mut().zip(chunk.chunks_exact(4)) {
                    *word = u32::from_le_bytes(wb.try_into().unwrap());
                }
                block
            })
            .collect();
        Ok(BloomPartial { blocks })
    }
}
