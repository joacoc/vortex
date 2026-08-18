// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ExecutionCtx;
use vortex_array::arrays::DecimalArray;
use vortex_array::match_each_decimal_value_type;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::BloomPartial;

pub(super) fn accumulate_decimal(
    array: &DecimalArray,
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

    /// Checks that only valid decimal values are added to the bloom filter.
    #[rstest]
    #[case(&[10i8, 20, 30, 40, 50])]
    fn validity_all_true<T>(#[case] present: &[T]) -> VortexResult<()>
    where
        T: Copy + Into<DecimalValue> + NativeDecimalType,
    {
        let ctx = setup()?;
        let decimal_dtype = DecimalDType::new(3, 0);
        let all_valid =
            DecimalArray::from_option_iter(present.iter().copied().map(Some), decimal_dtype);
        let bloom_filter = build_filter(
            all_valid.into_array(),
            DType::Decimal(decimal_dtype, Nullability::Nullable),
            ctx,
        )?;

        for &v in present {
            let scalar = Scalar::decimal(v.into(), decimal_dtype, Nullability::Nullable);
            assert!(bloom_filter.contains_valid_scalar(&scalar)?);
        }

        Ok(())
    }

    #[rstest]
    #[case(&[10i8, 20, 30, 40, 50])]
    fn validity_all_false<T>(#[case] present: &[T]) -> VortexResult<()>
    where
        T: Copy + Into<DecimalValue> + NativeDecimalType,
    {
        let ctx = setup()?;
        let decimal_dtype = DecimalDType::new(3, 0);
        let all_invalid =
            DecimalArray::from_option_iter(present.iter().map(|_| None::<T>), decimal_dtype);
        let bloom_filter = build_filter(
            all_invalid.into_array(),
            DType::Decimal(decimal_dtype, Nullability::Nullable),
            ctx,
        )?;

        for &v in present {
            let scalar = Scalar::decimal(v.into(), decimal_dtype, Nullability::Nullable);
            assert!(!bloom_filter.contains_valid_scalar(&scalar)?);
        }

        Ok(())
    }

    #[rstest]
    #[case(&[10i8, 20, 30, 40, 50])]
    fn validity_mixed<T>(#[case] present: &[T]) -> VortexResult<()>
    where
        T: Copy + Into<DecimalValue> + NativeDecimalType,
    {
        let ctx = setup()?;
        let decimal_dtype = DecimalDType::new(3, 0);
        let mixed = present
            .iter()
            .enumerate()
            .map(|(i, &v)| if i % 2 == 0 { Some(v) } else { None });
        let bloom_filter = build_filter(
            DecimalArray::from_option_iter(mixed, decimal_dtype).into_array(),
            DType::Decimal(decimal_dtype, Nullability::Nullable),
            ctx,
        )?;

        for (i, &v) in present.iter().enumerate() {
            let scalar = Scalar::decimal(v.into(), decimal_dtype, Nullability::Nullable);
            assert_eq!(bloom_filter.contains_valid_scalar(&scalar)?, i % 2 == 0,);
        }

        Ok(())
    }
}
