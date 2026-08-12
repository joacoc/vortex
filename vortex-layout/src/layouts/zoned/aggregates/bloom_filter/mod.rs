//! Bloom-filter aggregate for zoned layouts.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::num::NonZeroUsize;

use vortex_array::ArrayRef;
use vortex_array::Columnar;
use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::AggregateFnId;
use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

mod canonical;
mod partial;

pub(in crate::layouts::zoned) mod constant;
pub use partial::BloomPartial;

use crate::layouts::zoned::skip_index::bloom::is_bloom_valid_dtype;

// (joacoc) Opted for blocks_count as the main way to tune
// the Bloom filter, but there are also other ways, like
// setting the false-positive probability (FPP) and
// number of distinct values (NDV). Given the statistics
// collected from a random subset of values, they could be
// used to determine the number of blocks automatically,
// which I think is more aligned with how Vortex works with
// encoders. But for now, this is a simple approach that
// pushes the block-count selection back to the user
// and otherwise defaults to [DEFAULT_BLOCKS_COUNT].
//
// In case that the blocks count stays as is,
// a guide on how to select the number of blocks
// would be great.

/// Bloom-filter tuning persisted as aggregate metadata.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BloomOptions {
    /// Number of blocks in the split block Bloom filter (SBBF).
    ///
    /// The filter is partitioned into 256-bit blocks. More blocks reduce the
    /// number of distinct values assigned to each block, reducing the
    /// false-positive rate at the cost of increased filter size.
    ///
    /// Defaults to: [DEFAULT_BLOCKS_COUNT].
    ///
    /// ### Block size and memory usage
    ///
    /// As a reference, you can use the following table
    /// to understand the relationship between block size and memory usage
    /// for a **single zone**.
    ///
    /// | `blocks_count` |      Memory |
    /// | --------------: | ----------: |
    /// |               8 |   **256 B** |
    /// |             256 |   **8 KiB** |
    /// |            8192 | **256 KiB** |
    /// |          65,536 |   **2 MiB** |
    /// |       1,048,576 |  **32 MiB** |
    blocks_count: NonZeroUsize,
}

impl BloomOptions {
    pub fn new(blocks_count: NonZeroUsize) -> Self {
        Self { blocks_count }
    }

    pub fn blocks(&self) -> NonZeroUsize {
        self.blocks_count
    }
}

/// The default value is derived from the default [WriteStrategyBuilder::row_block_size]
const DEFAULT_BLOCKS_COUNT: usize = 256;

impl Default for BloomOptions {
    fn default() -> Self {
        Self {
            blocks_count: NonZeroUsize::new(DEFAULT_BLOCKS_COUNT).expect("valid blocks size"),
        }
    }
}

impl Display for BloomOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "blocks={}", self.blocks_count)
    }
}

/// Aggregate that stores one fixed-size Bloom block as a `Binary` scalar for every zone.
/// The bloom filter only stores valid scalar values, and does not consider
/// `NULL` values as part of the filter. If you need to know if a zone has null values
/// you should use the NULL count.
///
/// Empty unit struct, in accordance to [AggregateFnVTable] definition.
#[derive(Clone, Debug)]
pub struct BloomFilter;

impl AggregateFnVTable for BloomFilter {
    type Options = BloomOptions;
    type Partial = BloomPartial;

    fn id(&self) -> AggregateFnId {
        static ID: CachedId = CachedId::new("vortex.bloom_filter.v1");
        *ID
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        let blocks = u32::try_from(options.blocks_count.get())?;
        let metadata = blocks.to_le_bytes().to_vec();
        Ok(Some(metadata))
    }

    fn deserialize(
        &self,
        metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        vortex_ensure_eq!(metadata.len(), 4, "invalid bloom metadata length");
        let blocks = u32::from_le_bytes([metadata[0], metadata[1], metadata[2], metadata[3]]);
        Ok(BloomOptions::new(
            NonZeroUsize::new(blocks as usize)
                .ok_or_else(|| vortex_err!("bloom blocks length must be non-zero"))?,
        ))
    }

    /// Returns [Binary(Nullability::NonNullable)] when input [DType] is valid.
    ///
    /// The [BloomFilter] is serialized/deserialized as a sequence of bytes
    /// that represents the filter state.
    ///
    /// An empty filter rather than being NULL is represented
    /// by a zero-initialized byte sequence (`0x0..0`)
    fn return_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        is_bloom_valid_dtype(input_dtype).then_some(DType::Binary(Nullability::NonNullable))
    }

    fn partial_dtype(&self, options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        self.return_dtype(options, input_dtype)
    }

    /// Returns an empty Bloom filter with all blocks zero-initialized.
    fn empty_partial(&self, options: &Self::Options, _: &DType) -> VortexResult<Self::Partial> {
        Ok(BloomPartial {
            blocks: vec![[0u32; 8]; options.blocks_count.get()],
        })
    }

    // Combination happens by doing an OR between both filters bits
    fn combine_partials(&self, partial: &mut Self::Partial, other: Scalar) -> VortexResult<()> {
        if other.is_null() {
            return Ok(());
        }

        let bytes = other
            .as_binary()
            .value()
            .ok_or_else(|| vortex_err!("non-null bloom partial has no bytes"))?;

        let other = BloomPartial::try_from(bytes.as_slice())?;

        vortex_ensure_eq!(
            partial.blocks.len(),
            other.blocks.len(),
            "bloom partial block count mismatch — partials built with different blocks_count"
        );

        for (dst, src) in partial.blocks.iter_mut().zip(other.blocks.iter()) {
            for i in 0..8 {
                dst[i] |= src[i];
            }
        }

        Ok(())
    }

    /// Returns the non-nullable binary representation of a bloom filter
    ///
    /// Basically turns each block into a single byte sequence.
    fn to_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        let mut bytes = Vec::with_capacity(partial.blocks.len() * 8 * size_of::<u32>());
        bytes.extend(
            partial
                .blocks
                .iter()
                .flatten()
                .flat_map(|block_seq| block_seq.to_le_bytes()),
        );

        Ok(Scalar::binary(bytes, Nullability::NonNullable))
    }

    fn reset(&self, partial: &mut Self::Partial) {
        partial.blocks.fill([0; 8]);
    }

    fn is_saturated(&self, partial: &Self::Partial) -> bool {
        partial.blocks.iter().all(|byte| *byte == [u32::MAX; 8])
    }

    fn accumulate(
        &self,
        partial: &mut Self::Partial,
        batch: &Columnar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        match batch {
            Columnar::Constant(constant) => constant::accumulate_constant(constant, partial)?,
            Columnar::Canonical(canonical) => {
                canonical::accumulate_canonical(canonical, partial, ctx)?
            }
        }
        Ok(())
    }

    fn finalize(&self, partials: ArrayRef) -> VortexResult<ArrayRef> {
        Ok(partials)
    }

    fn finalize_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        self.to_scalar(partial)
    }
}

// The following functions are utils/useful for tests in canonical and constants.

#[cfg(test)]
pub(in crate::layouts::zoned::aggregates::bloom_filter) mod test_utils {
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::Accumulator;
    use vortex_array::aggregate_fn::DynAccumulator;
    use vortex_error::vortex_ensure;

    use super::*;

    pub fn setup() -> VortexResult<ExecutionCtx> {
        let session = vortex_array::array_session();
        let options = BloomOptions::default();
        let metadata = BloomFilter
            .serialize(&options)?
            .expect("bloom is serializable");
        assert_eq!(BloomFilter.deserialize(&metadata, &session)?, options);

        let ctx = session.create_execution_ctx();

        Ok(ctx)
    }

    pub fn extract_bloom_blocks(state: &Scalar) -> VortexResult<Vec<[u32; 8]>> {
        let bytes = state.as_binary().value().expect("bloom state is non-null");
        vortex_ensure!(
            bytes.len() % (8 * size_of::<u32>()) == 0,
            "invalid bloom state length"
        );
        let mut blocks = Vec::with_capacity(bytes.len() / 32);
        for block_bytes in bytes.chunks_exact(32) {
            let mut block = [0u32; 8];
            for (word, word_bytes) in block.iter_mut().zip(block_bytes.chunks_exact(4)) {
                *word = u32::from_le_bytes(
                    word_bytes
                        .try_into()
                        .expect("chunks_exact(4) always produces 4 bytes"),
                );
            }
            blocks.push(block);
        }
        Ok(blocks)
    }

    pub fn build_filter(
        batch: ArrayRef,
        dtype: DType,
        mut ctx: ExecutionCtx,
    ) -> VortexResult<BloomPartial> {
        let mut accumulator = Accumulator::try_new(BloomFilter, BloomOptions::default(), dtype)?;
        accumulator.accumulate(&batch.into_array(), &mut ctx)?;
        let state = accumulator.finish()?;
        let blocks = extract_bloom_blocks(&state)?;
        let bloom_filter = BloomPartial::from(blocks);

        Ok(bloom_filter)
    }

    #[test]
    fn saturation_false_when_empty() -> VortexResult<()> {
        let options = BloomOptions::default();
        let partial =
            BloomFilter.empty_partial(&options, &DType::Binary(Nullability::NonNullable))?;
        assert!(!BloomFilter.is_saturated(&partial));
        Ok(())
    }

    #[test]
    fn saturation_true_when_every_block_is_full() {
        let blocks = vec![[u32::MAX; 8]; 4];
        let partial = BloomPartial::from(blocks);

        assert!(BloomFilter.is_saturated(&partial));
    }

    #[test]
    fn combine_partials_rejects_mismatched_block_counts() -> VortexResult<()> {
        let mut smaller = BloomFilter.empty_partial(
            &BloomOptions::new(NonZeroUsize::new(4).unwrap()),
            &DType::Binary(Nullability::NonNullable),
        )?;
        let bigger = BloomFilter.empty_partial(
            &BloomOptions::default(),
            &DType::Binary(Nullability::NonNullable),
        )?;

        let bigger_scalar = BloomFilter.to_scalar(&bigger)?;
        let result = BloomFilter.combine_partials(&mut smaller, bigger_scalar);

        assert!(
            result.is_err(),
            "combining partials built with different blocks_count must fail loudly, not corrupt state"
        );
        Ok(())
    }

    #[test]
    fn combine_partials_unions_two_disjoint_partials() -> VortexResult<()> {
        let mut partial = BloomFilter.empty_partial(
            &BloomOptions::default(),
            &DType::Binary(Nullability::NonNullable),
        )?;
        for i in 0..50i64 {
            partial.insert(i);
        }

        let mut secondary_partial = BloomFilter.empty_partial(
            &BloomOptions::default(),
            &DType::Binary(Nullability::NonNullable),
        )?;
        for i in 50..100i64 {
            secondary_partial.insert(i);
        }

        // The following expected works because seed is equal for all.
        // If the seed is different for both partials, then this will fail.
        let mut expected = BloomFilter.empty_partial(
            &BloomOptions::default(),
            &DType::Binary(Nullability::NonNullable),
        )?;
        for i in 0..100i64 {
            expected.insert(i);
        }

        let secondary_partial_as_scalar = BloomFilter.to_scalar(&secondary_partial)?;
        BloomFilter.combine_partials(&mut partial, secondary_partial_as_scalar)?;

        assert_eq!(
            partial.blocks, expected.blocks,
            "merging via combine_partials should equal a single filter built from the union of inputs"
        );

        for i in 0..100i64 {
            let hash = partial.hash(i);
            assert!(partial.find_hash(hash), "value {i} missing after merge");
        }

        for i in 101..200i64 {
            let hash = partial.hash(i);
            assert!(
                !partial.find_hash(hash),
                "value {i} shouldn't be present after"
            );
        }

        Ok(())
    }
}
