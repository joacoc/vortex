// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Skipping-index interface and implementations.
//!
//! The skip index provides a way to prune zones
//! by using the results of an aggregation function.
//! During writes, an aggregation function calculates stats for each zone.
//! During reads, these stats are interpreted to determine whether a zone
//! can be pruned.
//!
//! Unlike a locating index, a skipping index summarizes a contiguous
//! range of rows in a zone. It does not locate matching rows. It only
//! proves that a zone cannot match a predicate.
//!
//! For correct reading and writing, a session must have registered the skip index.
//! If a skip index is not loaded and the session allows unknown plugins,
//! Vortex disables zoned pruning and scans the data normally. Otherwise,
//! opening the file fails.
//!
//! The skip index is coupled to the zoned layout through [`ZonedLayoutOptions`].
//! To set one up, a user must declare a skip index for a field
//! through the writer configuration.

use std::sync::Arc;

use vortex_array::aggregate_fn::AggregateFnId;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::SessionExt;
use vortex_session::VortexSession;

use super::writer::ZonedLayoutOptions;

/// One definition that supplies a persisted aggregate and registers every read-side component
/// needed to consult it.
///
/// The writer helper [`ZonedLayoutOptions::with_skip_index`] is the explicit per-column declaration
/// seam. Readers call [`SkipIndexSessionExt::register_skip_index`] on their
/// [`VortexSession`] before opening the file.
pub trait SkipIndex: Send + Sync + 'static {
    /// Returns the inner aggregate identifier.
    fn aggregate_id(&self) -> AggregateFnId;

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

    pub fn aggregate_id(&self) -> AggregateFnId {
        self.0.aggregate_id()
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
    pub fn with_skip_index(mut self, skip_index: SkipIndexRef) -> VortexResult<Self> {
        let mut skip_indexes = self
            .skip_indexes
            .take()
            .map(|indexes| indexes.to_vec())
            .unwrap_or_default();

        if skip_indexes
            .iter()
            .any(|index| index.aggregate_id() == skip_index.aggregate_id())
        {
            vortex_bail!(
                "skip index aggregate {} is already configured",
                skip_index.aggregate_id()
            );
        }

        skip_indexes.push(skip_index);
        self.skip_indexes = Some(skip_indexes.into());
        Ok(self)
    }
}
