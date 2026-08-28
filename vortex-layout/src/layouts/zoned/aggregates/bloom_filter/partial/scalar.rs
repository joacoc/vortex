// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scalar helpers for Split Block Bloom Filters (SBBFs).
//!
//! The core Bloom filter operates on bytes. This module provides scalar-aware
//! insertion and membership helpers, including validation and conversion from
//! [`Scalar`] values to the bytes used for hashing.

use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::dtype::ToBytes;
use vortex_array::match_each_float_ptype;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::BloomPartial;

/// The following implementation provides a simpler access for scalars.
impl BloomPartial {
    /// Returns the hash of the scalar's underlying value.
    /// Returns an error if the [Scalar] is invalid or its [DType] is unsupported.
    ///
    /// For example, `Scalar(Primitive(I32(54)))` is hashed as `hash(54)`.
    fn hash_valid_scalar(&self, scalar: &Scalar) -> VortexResult<u64> {
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
        self.add_hash(hash);

        Ok(())
    }

    /// A cleaner insert path for primitives that
    /// have the trait [ToBytes].
    #[inline]
    pub(in crate::layouts::zoned) fn insert_primitive<T>(&mut self, value: &T)
    where
        T: ToBytes,
    {
        let hash = self.hash(value.to_le_bytes());
        self.add_hash(hash);
    }
}
