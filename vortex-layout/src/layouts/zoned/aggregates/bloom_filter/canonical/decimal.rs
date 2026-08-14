// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::Array;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::Decimal;
use vortex_array::match_each_decimal_value_type;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::layouts::zoned::aggregates::bloom_filter::BloomPartial;

pub(super) fn accumulate_decimal(
    array: &Array<Decimal>,
    partial: &mut BloomPartial,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    match_each_decimal_value_type!(array.values_type(), |D| {
        match array.validity()?.execute_mask(array.len(), ctx)? {
            Mask::AllTrue(_) => {
                array
                    .buffer::<D>()
                    .iter()
                    .for_each(|value| partial.insert(value.to_le_bytes()));
            }
            Mask::AllFalse(_) => {}
            Mask::Values(v) => {
                array
                    .buffer::<D>()
                    .iter()
                    .zip(v.bit_buffer().iter())
                    .for_each(|(value, valid)| {
                        if valid {
                            partial.insert(value.to_le_bytes())
                        }
                    });
            }
        }
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::arrays::DecimalArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::DecimalDType;
    use vortex_array::dtype::NativeDecimalType;
    use vortex_array::dtype::Nullability;
    use vortex_array::scalar::DecimalValue;
    use vortex_array::scalar::Scalar;
    use vortex_error::VortexResult;

    use crate::layouts::zoned::aggregates::bloom_filter::test_utils::build_filter;
    use crate::layouts::zoned::aggregates::bloom_filter::test_utils::setup;

    #[rstest]
    #[case(3u8, 0i8, &[1i8, 2, 3], 99i8)]
    #[case(5u8, 1i8, &[10i16, 20, 30], 99i16)]
    #[case(9u8, 2i8, &[1000i32, 2000, 3000], 99999i32)]
    #[case(18u8, 2i8, &[1000i64, 2000, 3000], 99999i64)]
    #[case(10u8, 2i8, &[1000i128, 2000, 3000], 99999i128)]
    fn membership<T>(
        #[case] precision: u8,
        #[case] scale: i8,
        #[case] present: &[T],
        #[case] absent: T,
    ) -> VortexResult<()>
    where
        T: Copy + Into<DecimalValue> + NativeDecimalType,
    {
        let ctx = setup()?;
        let decimal_dtype = DecimalDType::new(precision, scale);
        let dtype = DType::Decimal(decimal_dtype, Nullability::NonNullable);
        let values: DecimalArray = DecimalArray::from_iter(present.iter().copied(), decimal_dtype);
        let bloom_filter = build_filter(values.into_array(), dtype, ctx)?;

        for &v in present {
            let scalar = Scalar::decimal(v.into(), decimal_dtype, Nullability::NonNullable);
            assert!(bloom_filter.contains_valid_scalar(&scalar)?);
        }

        let absent_scalar = Scalar::decimal(absent.into(), decimal_dtype, Nullability::NonNullable);
        assert!(!bloom_filter.contains_valid_scalar(&absent_scalar)?);

        Ok(())
    }
}
