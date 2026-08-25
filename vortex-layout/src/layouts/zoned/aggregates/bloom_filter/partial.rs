// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Split block Bloom filters (SBBF) implementation for Vortex.
//!
//! This implementation follows the original paper but
//! with the following noticeable changes:
//! - Renaming `bucket` to `block`,
//! - Small changes that help the Rust compiler generate optimized, vectorized
//!   code for `make_mask`, `add_hash`, and `find_hash`
//! - A different salt order.
//!
//! [Split block Bloom filters]: https://arxiv.org/pdf/2101.01719

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;

use twox_hash::XxHash3_64;
use twox_hash::XxHash64;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::dtype::ToBytes;
use vortex_array::match_each_float_ptype;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
#[cfg(test)]
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;

use super::BloomOptions;

const LANES_PER_BLOCK: usize = 8;
const BYTES_PER_LANE: usize = size_of::<u32>(); // 4 bytes

/// Block size (32 bytes [256 bits])
pub(super) const BLOCK_SIZE: usize = LANES_PER_BLOCK * BYTES_PER_LANE;

/// In the XXH family, the seed is optional and defaults to zero. Some crate
/// APIs, such as [`XxHash64`], require it to be supplied explicitly.
///
/// See the [XXH specification](https://github.com/Cyan4973/xxHash/blob/v0.8.3/doc/xxhash_spec.md#step-1-initialize-internal-accumulators).
const DEFAULT_SEED: u64 = 0;

/// Eight odd constants for multiply-shift hashing.
///
/// They fit in one 256-bit SIMD vector, and the order matches the one
/// used by the Apache Parquet specification. The paper's example uses
/// the same values but in a different order. This was not
/// intentional for having compatibility with Apache Parquet, but remains
/// as a common-order in implementations.
///
/// It is important to notice that while order doesn't affect validity,
/// it changes the final bits set in each lane.
const SALT: [u32; 8] = [
    0x47b6137b, 0x44974d91, 0x8824ad5b, 0xa2b7289d, 0x705495c7, 0x2df1424b, 0x9efc4947, 0x5c6bfb31,
];

/// Hash function to use in a bloom filter.
///
/// The current options are fast, non-cryptographic xxHash variants.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum HashFn {
    XxHash3_64 = 0, // Default
    XxHash64 = 1,
}

impl Display for HashFn {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::XxHash3_64 => "xxhash3_64",
            Self::XxHash64 => "xxhash64",
        })
    }
}

impl HashFn {
    /// Hashes the given bytes using the configured function.
    ///
    /// A match will pay a small branch cost in the hot path. A function pointer is
    /// slower, though, and a generic would require choosing the hash function
    /// at compile time (this is runtime configuration).
    ///
    /// There may be a way to move this match out of the hot path, but haven't
    /// figured it out yet.
    ///
    /// Would call it hash but clashes with [Hash] macro.
    #[inline]
    fn hash_bytes(self, bytes: &[u8]) -> u64 {
        match self {
            Self::XxHash3_64 => XxHash3_64::oneshot(bytes),
            Self::XxHash64 => XxHash64::oneshot(DEFAULT_SEED, bytes),
        }
    }
}

impl TryFrom<u32> for HashFn {
    type Error = VortexError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::XxHash3_64),
            1 => Ok(Self::XxHash64),
            _ => Err(vortex_err!("unknown bloom hash function ID: {value}")),
        }
    }
}

/// Represents a Split block Bloom Filter for a single layout zone.
pub struct BloomPartial {
    blocks: Vec<[u32; 8]>,
    hash_fn: HashFn,
}

impl BloomPartial {
    /// Returns the blocks len.
    ///
    /// Matches [BloomOptions::blocks_count]
    #[inline]
    pub(super) fn len(&self) -> usize {
        self.blocks.len()
    }

    #[inline]
    pub fn insert<T>(&mut self, value: T)
    where
        T: AsRef<[u8]>,
    {
        let hash = self.hash(value);
        self.insert_hash(hash);
    }

    #[inline]
    fn insert_hash(&mut self, hash: u64) {
        self.add_hash(hash);
    }

    /// Returns `true` if `value` might be present in the filter.
    ///
    /// A `false` result guarantees that the value is absent. A `true` result may
    /// be a false positive.
    ///
    /// Use `BloomPartial::contains_valid_scalar` for scalar values.
    #[inline]
    pub fn contains<T>(&self, value: T) -> bool
    where
        T: AsRef<[u8]>,
    {
        let hash = self.hash(value);
        self.find_hash(hash)
    }

    /// Produces a 64-bit hash.
    ///
    /// This follows the reference implementation, where
    /// the upper 32 bits select the block and the lower 32 bits determine the bit
    /// positions within that block.
    #[inline]
    fn hash<T>(&self, value: T) -> u64
    where
        T: AsRef<[u8]>,
    {
        self.hash_fn.hash_bytes(value.as_ref())
    }

    /// Returns the lower 32 bits of the hash used to construct the block mask.
    #[inline]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the mask uses the low 32 bits of the 64-bit hash"
    )]
    fn lower_hash_bits(&self, hash: u64) -> u32 {
        hash as u32
    }

    fn add_hash(&mut self, hash: u64) {
        let block_idx = self.block_index(hash, self.blocks.len());
        let mask = self.make_mask(self.lower_hash_bits(hash));

        // The original solution uses _mm256_sllv_epi32
        // or the mask into the existing block
        for i in 0..8 {
            self.blocks[block_idx][i] |= mask[i];
        }
    }

    /// Checks whether a hash is (probably) present in the filter.
    fn find_hash(&self, hash: u64) -> bool {
        let idx = self.block_index(hash, self.blocks.len());
        let mask = self.make_mask(self.lower_hash_bits(hash));

        let mut missing = 0u32;
        let block = &self.blocks[idx];

        // The original solution uses _mm256_testc_si256
        // checks if all the bits in mask are also set in *block. Scalar
        // equivalent: (~block & mask) == 0
        for i in 0..8 {
            missing |= !block[i] & mask[i];
        }

        missing == 0
    }

    /// Takes a hash value and creates a mask with one bit set in each 32-bit lane.
    /// These are the bits to set or check when accessing the block.
    fn make_mask(&self, hash: u32) -> [u32; 8] {
        let mut out = [0u32; 8];

        for i in 0..8 {
            // Shift all data right, reducing the hash values from 32 bits to five bits.
            // Those five bits represent an index in [0, 31)
            let y = hash.wrapping_mul(SALT[i]) >> 27;

            // Set a bit in each lane based on using the [0, 32) data as shift values.
            out[i] = 1u32 << y;
        }

        out
    }

    /// Returns the index of the block to which a hash belongs.
    ///
    /// For details about the algorithm, see
    /// [Lemire's FastRange](https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/).
    ///
    /// Although `blocks_count` is a `usize`, its value is limited to `u32::MAX`
    /// by [`BloomOptions`] and the serialization format.
    #[inline]
    fn block_index(&self, hash: u64, blocks_count: usize) -> usize {
        (((hash >> 32) * blocks_count as u64) >> 32) as usize
    }
}

/// Practical implementation to avoid having to share blocks
impl BloomPartial {
    #[inline]
    pub(super) fn reset(&mut self) {
        self.blocks.fill([0; 8]);
    }

    #[inline]
    pub(super) fn is_saturated(&self) -> bool {
        self.blocks.iter().all(|byte| *byte == [u32::MAX; 8])
    }

    /// Merges a compatible serialized Bloom filter into this partial.
    ///
    /// The merge is a bitwise OR, which represents the union of two split-block
    /// Bloom filters when they use the same block count.
    ///
    /// _Notice_ This method only validates the byte length.
    /// Merging bytes from a filter created with a different hash function
    /// will produce an invalid filter and introduce false negatives.
    #[inline]
    pub(super) fn merge(&mut self, other: &[u8]) -> VortexResult<()> {
        // Partial returns size in blocks,
        // while bytes contains len in amount of bytes.
        // So blocks * block_size (bytes) = total amount of bytes
        vortex_ensure_eq!(
            self.len() * BLOCK_SIZE,
            other.len(),
            "bloom partial block count mismatch"
        );

        for (dst_block, src_block) in self
            .blocks
            .iter_mut()
            .zip(other.as_chunks::<BLOCK_SIZE>().0)
        {
            for (dst_lane, src_lane) in dst_block
                .iter_mut()
                .zip(src_block.as_chunks::<BYTES_PER_LANE>().0)
            {
                *dst_lane |= u32::from_le_bytes(*src_lane);
            }
        }

        Ok(())
    }
}

/// The following implementation provides a simpler access for scalars.
impl BloomPartial {
    /// Returns the hash of the scalar's underlying value.
    /// Returns an error if the [Scalar] is invalid or its [DType] is unsupported.
    ///
    /// For example, `Scalar(Primitive(I32(54)))` is hashed as `hash(54)`.
    pub fn hash_valid_scalar(&self, scalar: &Scalar) -> VortexResult<u64> {
        if scalar.is_null() {
            return Err(vortex_err!("cannot hash invalid scalars in bloom filter"));
        }

        Ok(match scalar.dtype() {
            DType::Extension(_) => {
                self.hash_valid_scalar(&scalar.as_extension().to_storage_scalar())?
            }
            DType::Bool(_) => self.hash([u8::from(
                scalar
                    .as_bool()
                    .value()
                    .vortex_expect("non-null boolean value"),
            )]),
            DType::Primitive(ptype, _) => match ptype {
                PType::F16 | PType::F32 | PType::F64 => {
                    match_each_float_ptype!(ptype, |T| {
                        let value = scalar
                            .as_primitive()
                            .typed_value::<T>()
                            .vortex_expect("non-null primitive value");
                        self.hash(value.to_le_bytes())
                    })
                }
                _ => match_each_integer_ptype!(ptype, |T| {
                    let value = scalar
                        .as_primitive()
                        .typed_value::<T>()
                        .vortex_expect("non-null primitive value");
                    self.hash(value.to_le_bytes())
                }),
            },
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
                return Err(vortex_err!(
                    "Unsupported scalar type for bloom filter: {other}"
                ));
            }
        })
    }

    /// Returns `true` if the underlying value of a [Scalar] might be present in the filter.
    ///
    /// A `false` result guarantees that the value is absent. A `true` result may
    /// be a false positive.
    ///
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
        self.insert_hash(hash);

        Ok(())
    }

    /// A cleanear insert path for primitives that
    /// have the trait [ToBytes].
    #[inline]
    pub(super) fn insert_primitive<T>(&mut self, value: &T)
    where
        T: ToBytes,
    {
        let hash = self.hash(value.to_le_bytes());
        self.insert_hash(hash);
    }
}

// Useful convertion implementations for serialization
// used in [`BloomPartial::to_scalar`]
impl From<&BloomPartial> for Vec<u8> {
    fn from(val: &BloomPartial) -> Self {
        let mut bytes = Vec::with_capacity(val.len() * BLOCK_SIZE);
        bytes.extend(
            val.blocks
                .iter()
                .flatten()
                .flat_map(|block_seq| block_seq.to_le_bytes()),
        );

        bytes
    }
}

impl From<&BloomOptions> for BloomPartial {
    fn from(options: &BloomOptions) -> Self {
        Self {
            blocks: vec![[0u32; 8]; options.blocks_count.get() as usize],
            hash_fn: options.hash_fn,
        }
    }
}

impl PartialEq for BloomPartial {
    fn eq(&self, other: &Self) -> bool {
        // Even if two partials have the same blocks,
        // a different hash function means they represent
        // different sets of values.
        self.blocks == other.blocks && self.hash_fn == other.hash_fn
    }
}

#[cfg(test)]
impl From<Vec<[u32; 8]>> for BloomPartial {
    fn from(value: Vec<[u32; 8]>) -> Self {
        BloomPartial {
            blocks: value,
            hash_fn: HashFn::XxHash3_64,
        }
    }
}

#[cfg(test)]
impl TryFrom<&[u8]> for BloomPartial {
    type Error = VortexError;

    /// Reconstructs a partial from its serialized byte representation
    /// (the same layout produced by `to_scalar`).
    fn try_from(bytes: &[u8]) -> VortexResult<Self> {
        vortex_ensure!(
            !bytes.is_empty() && bytes.len().is_multiple_of(BLOCK_SIZE),
            "invalid bloom filter byte length: {}",
            bytes.len()
        );

        let blocks = bytes
            .as_chunks::<BLOCK_SIZE>()
            .0
            .iter()
            .map(|chunk| {
                let (lane_bytes, remainder) = chunk.as_chunks::<BYTES_PER_LANE>();
                let mut block = [0u32; 8];
                vortex_ensure!(
                    remainder.is_empty(),
                    "invalid bloom filter, unexpected remainder bytes"
                );

                for (lane, lane_bytes) in block.iter_mut().zip(lane_bytes) {
                    *lane = u32::from_le_bytes(*lane_bytes);
                }

                Ok(block)
            })
            .collect::<VortexResult<Vec<_>>>()?;

        vortex_ensure!(
            !blocks.is_empty() && u32::try_from(blocks.len()).is_ok(),
            "bloom blocks length must be non-zero and lower than u32::MAX",
        );

        Ok(BloomPartial {
            blocks,
            hash_fn: HashFn::XxHash3_64,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use rstest::rstest;
    use vortex_array::dtype::ToBytes;

    use crate::layouts::zoned::aggregates::bloom_filter::BloomOptions;
    use crate::layouts::zoned::aggregates::bloom_filter::BloomPartial;
    use crate::layouts::zoned::aggregates::bloom_filter::DEFAULT_BLOCKS_COUNT;
    use crate::layouts::zoned::aggregates::bloom_filter::HashFn;

    #[test]
    fn equality_includes_hash_function() {
        let blocks = vec![[0u32; 8]];
        let xxhash3 = BloomPartial {
            blocks: blocks.clone(),
            hash_fn: HashFn::XxHash3_64,
        };
        let xxhash64 = BloomPartial {
            blocks,
            hash_fn: HashFn::XxHash64,
        };

        assert!(xxhash3 != xxhash64);
    }

    #[test]
    fn bigger_filter_size() {
        // The idea is to create a bigger Bloom filter than the default one (1000x approx. ~8MiB),
        // but also so big that the chance of a false positive is not zero but is very low.
        // For this fixed set of values ([1, 1M]), as of this writing,
        // no false positives appear.
        //
        // The filter contains only even numbers. Finding an odd number would be a
        // valid Bloom-filter false positive, but because this test currently has none,
        // seeing one in the future would mean that something changed and should be reviewed.
        //
        // A change in how hashes are calculated or how the block index is selected
        // could trigger this assertion, like a kind of smoke test.
        let options = BloomOptions::new(
            NonZeroU32::new(DEFAULT_BLOCKS_COUNT * 1000).expect("valid nonzero u32"),
            HashFn::XxHash3_64,
        );
        let mut bloom_filter = BloomPartial::from(&options);

        for i in 1..=1_000_000u64 {
            if i % 2 == 0 {
                bloom_filter.insert(i.to_le_bytes());
            }
        }

        for i in 1..=1_000_000u64 {
            if i % 2 == 0 {
                assert!(
                    bloom_filter.contains(i.to_le_bytes()),
                    "expected {i} to exist"
                );
            } else {
                assert!(
                    !bloom_filter.contains(i.to_le_bytes()),
                    "unexpected false positive for odd number {i}"
                );
            }
        }
    }

    #[test]
    fn valid_serde() {
        let mut bloom_filter = BloomPartial::from(&BloomOptions::default());
        bloom_filter.insert(32.to_le_bytes());

        let bytes: Vec<u8> = (&bloom_filter).into();
        let valid_filter = BloomPartial::try_from(bytes.as_slice()).unwrap();

        assert!(
            valid_filter.contains(32.to_le_bytes()),
            "expect filter to have value"
        );

        assert!(
            !valid_filter.contains(14.to_le_bytes()),
            "expect filter to not have value"
        );
    }

    #[test]
    fn invalid_serde() {
        let mut bloom_filter = BloomPartial::from(&BloomOptions::default());
        bloom_filter.insert(32.to_le_bytes());

        let mut bytes: Vec<u8> = (&bloom_filter).into();
        bytes.pop();
        let invalid_filter = BloomPartial::try_from(bytes.as_slice());

        assert!(invalid_filter.is_err(), "expect filter to be invalid");

        let mut bytes: Vec<u8> = (&bloom_filter).into();
        bytes.push(0u8);
        let invalid_filter = BloomPartial::try_from(bytes.as_slice());

        assert!(invalid_filter.is_err(), "expect filter to be invalid");
    }

    /// Another regression test for bloom serialization,
    /// but in this case to detect mask salt changes.
    /// It just verifies that a filter's serialized representation remains stable.
    #[test]
    fn serialized_bits_are_stable() {
        let options = BloomOptions::new(NonZeroU32::MIN, HashFn::XxHash3_64);
        let mut bloom_filter = BloomPartial::from(&options);

        bloom_filter.insert(b"vortex");

        // Because we have only one block, and  this is the only value inserted,
        // these lanes equal its mask: `empty | mask == mask`.
        let expected_lanes: [u32; 8] = [
            0x0000_1000,
            0x0200_0000,
            0x0000_2000,
            0x0800_0000,
            0x0200_0000,
            0x0000_0040,
            0x0000_4000,
            0x0000_1000,
        ];

        let expected_bytes: Vec<u8> = expected_lanes
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();

        let bytes: Vec<u8> = (&bloom_filter).into();
        assert_eq!(bytes, expected_bytes);
    }

    // Similar to the goldenfile tests, but for hash functions.
    //
    // Useful compatibility test to catch an accidental hash-algorithm or seed change.
    #[rstest]
    #[case(HashFn::XxHash3_64, 16649171463689419262)]
    #[case(HashFn::XxHash64, 631098470869724288)]
    fn hash_output_is_stable(#[case] hash_fn: HashFn, #[case] expected: u64) {
        let mut bloom_filter = BloomPartial::from(&BloomOptions::new(
            NonZeroU32::new(256).expect("valid non-zero"),
            hash_fn,
        ));
        assert_eq!(bloom_filter.hash(b"vortex"), expected);

        // Additional check
        bloom_filter.insert(b"vortex");
        assert!(bloom_filter.contains(b"vortex"));
    }
}
