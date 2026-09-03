// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Skipping-index interface and implementations.
//!
//! The skip index provides a way to prune zones by using the results
//! of an aggregation function.
//! During writes, the aggregation function calculates stats for each zone.
//! During reads, these stats are interpreted to determine whether a zone
//! can be pruned.
//!
//! The skip index is coupled to the zoned layout through [`ZonedLayoutOptions`].
//! To set one up, a user must declare a skip index for a field
//! through the writer configuration.
//!
//! ### Difference from a locating index
//!
//! Unlike a locating index, a skipping index summarizes a zone.
//! It does not locate matching rows. It can only prove that a zone
//! cannot match a predicate.

use std::sync::Arc;

use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::dtype::DType;
use vortex_session::SessionExt;
use vortex_session::VortexSession;

use super::writer::ZonedLayoutOptions;

/// One definition that supplies a persisted aggregate and registers every read-side component
/// needed to consult it.
///
/// The writer helper [`ZonedLayoutOptions::with_skip_index`] is the explicit per-column declaration
/// seam. Readers call [`SkipIndexSessionExt::register_skip_index`] on their
/// [`VortexSession`] before opening the file.
///
/// ## Logical and physical representation
///
/// [`SkipIndex`] defines the logical skip-index abstraction. A skip index has no identifier
/// or serialized representation of its own. Instead, it is represented in a Vortex file by
/// the serialized [`AggregateFnRef`] it provides.
pub trait SkipIndex: Send + Sync + 'static {
    /// The aggregate state to persist for `input_dtype`, or `None` when unsupported.
    fn aggregate_fn(&self, input_dtype: &DType) -> Option<AggregateFnRef>;

    /// Register the aggregate, optional probe function, and predicate rewrite as one operation.
    fn register(&self, session: &VortexSession);
}

/// Extension trait for registering skipping indexes with a Vortex session.
pub trait SkipIndexSessionExt: SessionExt {
    /// Registers a skip index with this session.
    ///
    /// Registration makes the skipping-index implementation available to the session,
    /// but does not add it to a file. To write one, add it to  [`ZonedLayoutOptions::with_skip_index`]
    /// when configuring the field's zoned layout in the `WriteStrategyBuilder`.
    ///
    /// # Usage example
    ///
    /// ```
    /// use vortex_layout::layouts::zoned::skip_index::{
    ///     SkipIndexRef, SkipIndexSessionExt,
    /// };
    /// use vortex_session::VortexSession;
    ///
    /// fn register_index(session: &VortexSession, index: &SkipIndexRef) {
    ///     session.register_skip_index(index);
    /// }
    /// ```
    fn register_skip_index(&self, skip_index: &SkipIndexRef) {
        skip_index.register(&self.session());
    }
}

impl<S: SessionExt> SkipIndexSessionExt for S {}

/// A reference-counted [`SkipIndex`].
#[derive(Clone)]
pub struct SkipIndexRef(Arc<dyn SkipIndex>);

impl SkipIndexRef {
    pub fn new(index_ref: Arc<dyn SkipIndex>) -> Self {
        SkipIndexRef(index_ref)
    }

    pub fn aggregate_fn(&self, input_dtype: &DType) -> Option<AggregateFnRef> {
        self.0.aggregate_fn(input_dtype)
    }

    pub fn register(&self, session: &VortexSession) {
        self.0.register(session);
    }
}

impl ZonedLayoutOptions {
    /// Add `skip_index` to this zoned writer while retaining the default min/max-style aggregates.
    ///
    /// `WriteStrategyBuilder::with_field_zoned_options` can install the configured options for one
    /// field while retaining the default data layout pipeline.
    pub fn with_skip_index(mut self, skip_index: SkipIndexRef) -> Self {
        let mut skip_indexes = self
            .skip_indexes
            .take()
            .map(|indexes| indexes.to_vec())
            .unwrap_or_default();

        skip_indexes.push(skip_index);
        self.skip_indexes = Some(skip_indexes.into());

        self
    }
}
