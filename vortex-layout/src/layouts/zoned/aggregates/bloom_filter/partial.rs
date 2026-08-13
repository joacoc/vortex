//! Split block Bloom filters (SBBF) implementation for vortex.
//!
//! [Split block Bloom filters]: https://arxiv.org/pdf/2101.01719

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;
use std::hash::Hasher;

use twox_hash::XxHash3_64;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::match_each_float_ptype;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::DecimalValue;
use vortex_array::scalar::Scalar;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

/// Block size in bits (8 * 4 = 32 bits)
pub(super) const BLOCK_SIZE: usize = 8 * size_of::<u32>();

/// Represents a Split block Bloom Filter filter for a single layout zone.
pub struct BloomPartial {
    pub(super) blocks: Vec<[u32; 8]>,
}

/// The following Split block Bloom filter (SBBF) implementation
/// is a translation of the original paper's names and values,
/// with slight changes to let the Rust compiler generate
/// optimized, vectorized code for `make_mask`, `add_hash`, and `find_hash`.
impl BloomPartial {
    /// Returns the blocks len.
    ///
    /// Matches [BloomOptions::blocks_count]
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
    fn insert_hash(&mut self, hash: u64) {
        self.add_hash(hash);
    }

    #[inline]
    fn hash<T>(&self, value: T) -> u64
    where
        T: Hash,
    {
        // > Since the seed is optional, it can be 0.
        // Ref: https://github.com/Cyan4973/xxHash/blob/v0.8.3/doc/xxhash_spec.md#step-1-initialize-internal-accumulators
        let mut hasher = XxHash3_64::with_seed(0);
        value.hash(&mut hasher);
        hasher.finish()
    }

    fn add_hash(&mut self, hash: u64) {
        let idx = self.block_index(hash, self.blocks.len()) as usize;

        let mask = self.make_mask(hash as u32);
        for i in 0..8 {
            self.blocks[idx][i] |= mask[i];
        }
    }

    /// Checks whether a hash is (probably) present in the filter.
    fn find_hash(&self, hash: u64) -> bool {
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

/// The following implementation provides a simpler access for scalars.
impl BloomPartial {
    /// Returns the hash of the scalar's underlying value.
    /// Returns an error if the [Scalar] is invalid or its [DType] is unsupported.
    ///
    /// For example, `Scalar(Primitive(I32(54)))` is hashed as `hash(54)`.
    pub(in crate::layouts::zoned) fn hash_valid_scalar(
        &self,
        scalar: &Scalar,
    ) -> VortexResult<u64> {
        if scalar.is_null() {
            return Err(vortex_err!("cannot hash invalid scalars in bloom filter"));
        }

        Ok(match scalar.dtype() {
            DType::Extension(_) => {
                self.hash_valid_scalar(&scalar.as_extension().to_storage_scalar())?
            }
            DType::Bool(_) => self.hash(
                scalar
                    .as_bool()
                    .value()
                    .vortex_expect("non-null boolean value"),
            ),
            DType::Primitive(ptype, _) => match ptype {
                PType::F16 | PType::F32 | PType::F64 => {
                    match_each_float_ptype!(ptype, |T| {
                        let value = scalar
                            .as_primitive()
                            .typed_value::<T>()
                            .vortex_expect("non-null primitive value");
                        self.hash(value.to_bits())
                    })
                }
                _ => match_each_integer_ptype!(ptype, |T| {
                    let value = scalar
                        .as_primitive()
                        .typed_value::<T>()
                        .vortex_expect("non-null primitive value");
                    self.hash(value)
                }),
            },
            DType::Decimal(..) => {
                let decimal = scalar
                    .as_decimal()
                    .decimal_value()
                    .vortex_expect("non-null decimal value");
                match decimal {
                    DecimalValue::I8(v) => self.hash(v),
                    DecimalValue::I16(v) => self.hash(v),
                    DecimalValue::I32(v) => self.hash(v),
                    DecimalValue::I64(v) => self.hash(v),
                    DecimalValue::I128(v) => self.hash(v),
                    DecimalValue::I256(v) => self.hash(v),
                }
            }
            DType::Utf8(_) => {
                let buffer = scalar
                    .as_utf8()
                    .value()
                    .vortex_expect("non-null utf8 value");
                self.hash(buffer.as_bytes())
            }
            DType::Binary(_) => {
                let buffer = scalar
                    .as_binary()
                    .value()
                    .vortex_expect("non-null binary value");
                self.hash(buffer.as_slice())
            }
            other => {
                return Err(vortex_err!("bloom filter does not support dtype {other}"));
            }
        })
    }

    /// Returns true if the underlying value of a [Scalar] may be present.
    /// Returns an error if the [Scalar] is invalid or its [DType] is unsupported.
    pub(in crate::layouts::zoned) fn contains_valid_scalar(
        &self,
        scalar: &Scalar,
    ) -> VortexResult<bool> {
        let hash = self.hash_valid_scalar(scalar)?;
        Ok(self.find_hash(hash))
    }

    /// Inserts the underlying value of a [Scalar] if it is valid.
    /// Returns an error if the [Scalar] is invalid or its [DType] is unsupported.
    pub(in crate::layouts::zoned) fn insert_valid_scalar(
        &mut self,
        scalar: &Scalar,
    ) -> VortexResult<()> {
        let hash = self.hash_valid_scalar(scalar)?;
        Ok(self.insert_hash(hash))
    }
}

#[cfg(test)]
impl BloomPartial {
    #[inline]
    pub(super) fn contains<T>(&self, value: T) -> bool
    where
        T: Hash,
    {
        let hash = self.hash(value);
        self.find_hash(hash)
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
                for (lane, lane_bytes) in block.iter_mut().zip(chunk.chunks_exact(4)) {
                    *lane = u32::from_le_bytes(lane_bytes.try_into().map_err(|_| {
                        vortex_err!("invalid bloom filter word length: {}", lane_bytes.len())
                    })?);
                }
                Ok(block)
            })
            .collect::<VortexResult<Vec<_>>>()?;

        Ok(BloomPartial { blocks })
    }
}
