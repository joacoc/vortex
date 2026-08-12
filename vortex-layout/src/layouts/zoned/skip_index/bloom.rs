//! Bloom skipping index for equality predicates.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::varbinview::VarBinViewArrayExt;
use vortex_array::dtype::DType;
use vortex_array::expr::Expression;
use vortex_array::expr::is_root;
use vortex_array::expr::not;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::literal::Literal;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::scalar_fn::session::ScalarFnSessionExt;
use vortex_array::stats::rewrite::StatsRewriteCtx;
use vortex_array::stats::rewrite::StatsRewriteRule;
use vortex_array::stats::session::StatsSessionExt;
use vortex_array::stats::stat;
use vortex_buffer::BitBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use super::SkipIndex;
pub use crate::layouts::zoned::aggregates::bloom_filter::BloomFilter;
pub use crate::layouts::zoned::aggregates::bloom_filter::BloomOptions;
pub use crate::layouts::zoned::aggregates::bloom_filter::BloomPartial;

/// Bloom skip index for constant-equality predicates.
///
/// TODO(joacoc): Add documentation about the Bloom skip index
/// and how it works here.
#[derive(Clone, Debug, Default)]
pub struct BloomSkipIndex {
    options: BloomOptions,
}

impl BloomSkipIndex {
    pub fn new(options: BloomOptions) -> Self {
        Self { options }
    }

    pub fn options(&self) -> &BloomOptions {
        &self.options
    }
}

impl SkipIndex for BloomSkipIndex {
    fn aggregate_fn(&self, input_dtype: &DType) -> Option<AggregateFnRef> {
        BloomFilter
            .return_dtype(&self.options, input_dtype)
            .map(|_| BloomFilter.bind(self.options.clone()))
    }

    fn register(&self, session: &VortexSession) {
        session.aggregate_fns().register(BloomFilter);
        session.scalar_fns().register(BloomContains);
        session.stats().register_rewrite(BloomEqRewrite {
            options: self.options.clone(),
        });
    }
}

/// Probe scalar function: test one literal against each binary Bloom state.
#[derive(Clone, Debug)]
struct BloomContains;

impl ScalarFnVTable for BloomContains {
    type Options = BloomOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.bloom_contains.v1");
        *ID
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        BloomFilter.serialize(options)
    }

    fn deserialize(&self, metadata: &[u8], session: &VortexSession) -> VortexResult<Self::Options> {
        BloomFilter.deserialize(metadata, session)
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(2)
    }

    /// Only two children are expected.
    /// The first child represents the filter as a byte sequence,
    /// while the second child represents the literal value to search for (the needle).
    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("filter"),
            1 => ChildName::from("needle"),
            _ => unreachable!("bloom_contains has exactly two children"),
        }
    }

    fn return_dtype(&self, _options: &Self::Options, args: &[DType]) -> VortexResult<DType> {
        vortex_ensure!(
            matches!(args[0], DType::Binary(_)),
            "bloom filter must be Binary"
        );
        vortex_ensure!(
            is_bloom_valid_dtype(&args[1]),
            "bloom needle must be bool, primitive, decimal, utf8, binary or extension"
        );

        Ok(DType::Bool(args[0].nullability() | args[1].nullability()))
    }

    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let filters = args.get(0)?.execute::<VarBinViewArray>(ctx)?;

        // If the needle accepts an array of values, e.g., Array[1, 2, 3],
        // the following code should be updated.
        let needle_array = args.get(1)?;
        let needle = needle_array
            .as_constant()
            .ok_or_else(|| vortex_err!("bloom needle must be constant"))?;

        let validity = filters.varbinview_validity();
        let valid = validity.execute_mask(filters.len(), ctx)?;

        // Quick return if the needle is invalid.
        if !needle.is_valid() {
            let possible = vec![false; filters.len()];
            return Ok(BoolArray::new(BitBuffer::from_iter(possible), validity).into_array());
        }

        let mut possible = Vec::with_capacity(filters.len());
        for (idx, is_valid) in valid.iter().enumerate() {
            if !is_valid {
                possible.push(false);
                continue;
            }

            let bytes = filters.bytes_at(idx);
            let partial = BloomPartial::try_from(bytes.as_slice())?;

            vortex_ensure!(
                partial.len() == options.blocks().get(),
                "stored bloom length does not match options"
            );

            possible.push(partial.contains_valid_scalar(&needle)?);
        }

        Ok(BoolArray::new(BitBuffer::from_iter(possible), validity).into_array())
    }

    fn is_null_sensitive(&self, _options: &Self::Options) -> bool {
        false
    }

    fn is_fallible(&self, _options: &Self::Options) -> bool {
        false
    }
}

/// Equality rewrite that turns a Bloom miss into a zone falsifier.
#[derive(Clone, Debug)]
struct BloomEqRewrite {
    options: BloomOptions,
}

impl StatsRewriteRule for BloomEqRewrite {
    fn scalar_fn_id(&self) -> ScalarFnId {
        Binary.id()
    }

    /// Only works for root literal comparisons and valid [DTypes].
    ///
    /// E.g. `eq(root(), lit(5i32))` or `eq(lit(5i32), root())`
    fn falsify(
        &self,
        expr: &Expression,
        ctx: &StatsRewriteCtx<'_>,
    ) -> VortexResult<Option<Expression>> {
        if *expr.as_::<Binary>() != Operator::Eq {
            return Ok(None);
        }

        let (column, literal) = if is_root(expr.child(0)) && expr.child(1).is::<Literal>() {
            (expr.child(0), expr.child(1))
        } else if is_root(expr.child(1)) && expr.child(0).is::<Literal>() {
            (expr.child(1), expr.child(0))
        } else {
            return Ok(None);
        };

        if !is_bloom_valid_dtype(&ctx.return_dtype(column)?) || literal.as_::<Literal>().is_null() {
            return Ok(None);
        }

        let filter = stat(column.clone(), BloomFilter.bind(self.options.clone()));
        let contains = BloomContains.new_expr(self.options.clone(), [filter, literal.clone()]);
        Ok(Some(not(contains)))
    }
}

/// Returns true if the type is valid for the bloom index to acc/contain.
///
/// This is defined by the available implementations in
/// [crate::layouts::zoned::aggregates::bloom::constant] and
/// [crate::layouts::zoned::aggregates::bloom::canonical]
pub(in crate::layouts::zoned) fn is_bloom_valid_dtype(dtype: &DType) -> bool {
    match dtype {
        DType::Extension(ext) => is_bloom_valid_dtype(ext.storage_dtype()),
        DType::Bool(_)
        | DType::Primitive(..)
        | DType::Decimal(..)
        | DType::Utf8(_)
        | DType::Binary(_) => true,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::Canonical;
    use vortex_array::Columnar;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::AggregateFnVTable;
    use vortex_array::aggregate_fn::AggregateFnVTableExt;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::expr::Expression;
    use vortex_array::expr::eq;
    use vortex_array::expr::gt_eq;
    use vortex_array::expr::lit;
    use vortex_array::expr::root;
    use vortex_array::scalar::Scalar;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use super::BloomSkipIndex;
    use super::SkipIndex;
    use crate::layouts::zoned::aggregates::bloom_filter::BloomFilter;
    use crate::layouts::zoned::aggregates::bloom_filter::BloomOptions;
    use crate::layouts::zoned::zone_map::ZoneMap;
    use crate::test::SESSION;

    #[test]
    fn missing_stat_stays_inconclusive() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let index = BloomSkipIndex::new(BloomOptions::default());
        index.register(&session);
        let predicate = eq(root(), lit(42i64));
        let proof = predicate
            .falsify(
                &DType::Primitive(PType::I64, Nullability::NonNullable),
                &session,
            )?
            .expect("equality has a bloom proof");

        let zone_map = ZoneMap::try_new(
            DType::Primitive(PType::I64, Nullability::NonNullable),
            StructArray::try_new(Vec::<&str>::new().into(), vec![], 2, Validity::NonNullable)?,
            Arc::new([]),
            8,
            16,
        )?;
        assert!(zone_map.prune(&proof, &session)?.all_false());
        Ok(())
    }

    /// Similar zone map tests as the ones in [crate::layouts::zoned::tests]
    /// but using BloomFilter rather than max/min zones.
    fn build_bloom_zone_map(dtype: DType, batch: ArrayRef) -> ZoneMap {
        let bloom = BloomFilter;
        let options = BloomOptions::default();

        // If index is not registered there will be no warning,
        // but the index will return false for everything.
        //
        // (joacoc) should be considered a warning for missing aggregatefns?
        BloomSkipIndex::new(options.clone()).register(&SESSION);
        let mut ctx = SESSION.create_execution_ctx();

        let mut zone_filter = bloom.empty_partial(&options, &dtype).unwrap();
        bloom
            .accumulate(
                &mut zone_filter,
                &Columnar::Canonical(batch.execute::<Canonical>(&mut ctx).unwrap()),
                &mut ctx,
            )
            .unwrap();

        let zone_filter_as_scalar = bloom.to_scalar(&zone_filter).unwrap();
        let zone_filter_as_bytes = zone_filter_as_scalar.as_binary().value().unwrap().to_vec();
        let zone_filter_as_varbin =
            VarBinArray::from_nullable_bytes(vec![Some(zone_filter_as_bytes.as_slice())]);

        let bloom = BloomFilter.bind(options);
        let zone_filter_struct = StructArray::from_fields(&[(
            bloom.clone().to_string(),
            zone_filter_as_varbin.into_array(),
        )])
        .unwrap();

        ZoneMap::try_new(dtype, zone_filter_struct, Arc::new([bloom]), 1, 10).unwrap()
    }

    fn assert_prune(zone_map: &ZoneMap, dtype: &DType, expr: Expression, expected: [bool; 1]) {
        let mut ctx = SESSION.create_execution_ctx();
        let pruning_expr = expr.falsify(dtype, &SESSION).unwrap().unwrap();
        let mask = zone_map.prune(&pruning_expr, &SESSION).unwrap();
        assert_arrays_eq!(mask.into_array(), BoolArray::from_iter(expected), &mut ctx);
    }

    #[rstest]
    #[case::equals_value_not_in_batch(eq(root(), lit(99i32)), [true])]
    #[case::equals_value_in_batch(eq(root(), lit(5i32)), [false])]
    #[case::gt_eq_not_supported_by_bloom(gt_eq(root(), lit(4i32)), [false])]
    #[case::null_never_prunes(
        gt_eq(root(), lit(Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)))),
        [false]
    )]
    fn test_zone_map_prunes_with_bloom_filter_i32(
        #[case] expr: Expression,
        #[case] expected: [bool; 1],
    ) -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::Nullable);
        let batch = PrimitiveArray::new(buffer![5i32, 6i32, 7i32], Validity::AllValid).into_array();
        let zone_map = build_bloom_zone_map(dtype.clone(), batch);
        assert_prune(&zone_map, &dtype, expr, expected);
        Ok(())
    }

    #[rstest]
    #[case::equals_value_not_in_batch(eq(root(), lit("zz")), [true])]
    #[case::equals_value_in_batch(eq(root(), lit("london")), [false])]
    fn test_zone_map_prunes_with_bloom_filter_varbin(
        #[case] expr: Expression,
        #[case] expected: [bool; 1],
    ) -> VortexResult<()> {
        let dtype = DType::Utf8(Nullability::NonNullable);
        let batch = VarBinArray::from_iter(
            [Some("london"), Some("hamburg"), Some("newyork")],
            dtype.clone(),
        )
        .into_array();
        let zone_map = build_bloom_zone_map(dtype.clone(), batch);
        assert_prune(&zone_map, &dtype, expr, expected);
        Ok(())
    }
}
