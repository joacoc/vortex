// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Aggregate functions selected by the zoned layout.

use std::sync::Arc;

use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::aggregate_fn::fns::nan_count::NanCount;
use vortex_array::aggregate_fn::fns::null_count::NullCount;
use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
use vortex_array::dtype::DType;
use vortex_session::VortexSession;

mod min_max;

use crate::layouts::zoned::aggregates::min_max::min_max_aggregate_fns;

pub mod bloom_filter;

pub(super) fn default_zoned_aggregate_fns(
    dtype: &DType,
    session: &VortexSession,
) -> Arc<[AggregateFnRef]> {
    let [max, min] = min_max_aggregate_fns(dtype);

    // Sum is deliberately absent: zone maps exist to prune, and a zone sum prunes nothing.
    // Its semantics are also unsettled - null-on-empty was changed in #9113 and reverted in
    // #9324 - so it is not a stat to record in every zone of every file, let alone freeze
    // into an edition. File-level statistics still record `Stat::Sum` via `PRUNING_STATS`.
    let mut aggregate_fns = vec![
        max,
        min,
        NanCount.bind(EmptyOptions),
        NullCount.bind(EmptyOptions),
    ];

    // Stats from spatial extension types are discovered from the registry at runtime instead.
    aggregate_fns.extend(session.aggregate_fns().zone_stat_defaults(dtype));

    aggregate_fns.into()
}

#[cfg(test)]
mod tests {
    use vortex_array::aggregate_fn::fns::sum::Sum;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::extension::datetime::TimeUnit;
    use vortex_array::extension::datetime::Timestamp;

    use super::default_zoned_aggregate_fns;

    // TODO (joacoc)
    // Uncomment after merging with BloomFilter code.
    //
    // use super::BloomFilter;
    // use super::bloom_filter::BloomOptions;
    // use super::default_zoned_aggregate_fns;
    //
    // #[test]
    // fn default_aggregates_exclude_bloom_filter() {
    //     let aggregate_fns =
    //         default_zoned_aggregate_fns(&PType::I64.into(), &vortex_array::array_session());
    //     let bloom = BloomFilter.bind(BloomOptions::default());

    //     assert!(
    //         aggregate_fns
    //             .iter()
    //             .all(|aggregate_fn| aggregate_fn != &bloom)
    //     );
    // }

    #[test]
    fn default_aggregates_skip_sum_for_non_summable_dtype() {
        let dtype = DType::Extension(
            Timestamp::new(TimeUnit::Microseconds, Nullability::Nullable).erased(),
        );
        let aggregate_fns = default_zoned_aggregate_fns(&dtype, &vortex_array::array_session());

        assert!(
            aggregate_fns
                .iter()
                .all(|aggregate_fn| !aggregate_fn.is::<Sum>())
        );
    }
}
