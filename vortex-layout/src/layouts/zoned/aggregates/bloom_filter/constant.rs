// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::ConstantArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::match_each_float_ptype;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::DecimalValue;
use vortex_array::scalar::Scalar;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::layouts::zoned::aggregates::bloom_filter::BloomPartial;

pub(super) fn accumulate_constant(
    constant: &ConstantArray,
    partial: &mut BloomPartial,
) -> VortexResult<()> {
    let scalar = constant.scalar();

    // Omit NULL values on purpose.
    if scalar.is_null() {
        return Ok(());
    }

    partial.insert_hash(partial.hash_valid_scalar(scalar)?);
    Ok(())
}

impl BloomPartial {
    /// Scalar values must be valid otherwise the function will raise an err.
    ///
    /// This function is used by both, for accumulating scalars,
    /// but also to get a scalar value membership.
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

    pub(in crate::layouts::zoned) fn contains_valid_scalar(
        &self,
        scalar: &Scalar,
    ) -> VortexResult<bool> {
        let hash = self.hash_valid_scalar(scalar)?;
        Ok(self.find_hash(hash))
    }
}

#[cfg(test)]
mod tests {

    use vortex_array::aggregate_fn::AggregateFnVTable;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::extension::datetime::TimeUnit;
    use vortex_array::extension::datetime::Timestamp;
    use vortex_array::scalar::Scalar;
    use vortex_error::VortexResult;

    use crate::layouts::zoned::aggregates::bloom_filter::BloomFilter;
    use crate::layouts::zoned::aggregates::bloom_filter::BloomOptions;
    use crate::layouts::zoned::aggregates::bloom_filter::constant::accumulate_constant;

    #[test]
    fn nulls_are_omitted() {
        let bloom = BloomFilter;
        let mut zone_partial = bloom
            .empty_partial(
                &BloomOptions::default(),
                &DType::Primitive(PType::I32, Nullability::Nullable),
            )
            .unwrap();

        assert!(
            accumulate_constant(
                &ConstantArray::new(
                    Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)),
                    1,
                ),
                &mut zone_partial,
            )
            .is_ok(),
            "expected to return Ok() for null scalars"
        )
    }

    #[test]
    fn null_raises_error_on_hash() {
        let bloom = BloomFilter;
        let zone_partial = bloom
            .empty_partial(
                &BloomOptions::default(),
                &DType::Primitive(PType::I32, Nullability::Nullable),
            )
            .unwrap();

        assert!(
            zone_partial
                .contains_valid_scalar(&Scalar::null(DType::Primitive(
                    PType::I32,
                    Nullability::Nullable
                )))
                .is_err(),
            "expected to return Err() for null scalars"
        )
    }

    #[test]
    fn valid_extension_is_a_member() -> VortexResult<()> {
        let ext_dtype = Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased();
        let scalar = Scalar::extension_ref(
            ext_dtype.clone(),
            Scalar::primitive(1_000i64, Nullability::NonNullable),
        );
        let bloom = BloomFilter;
        let mut zone_partial =
            bloom.empty_partial(&BloomOptions::default(), &DType::Extension(ext_dtype))?;

        accumulate_constant(&ConstantArray::new(scalar.clone(), 1), &mut zone_partial)?;

        assert!(zone_partial.contains_valid_scalar(&scalar)?);
        Ok(())
    }
}
