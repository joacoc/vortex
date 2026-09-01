// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Skipping-index interface and implementations.

use std::sync::Arc;

use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::dtype::DType;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use super::writer::ZonedLayoutOptions;

/// One definition that supplies a persisted aggregate and registers every read-side component
/// needed to consult it.
///
/// The writer helper [`ZonedLayoutOptions::with_skip_index`] is the explicit per-column declaration
/// seam. Readers call [`SkipIndex::register`] on their session before opening the file.
pub trait SkipIndex: Send + Sync + 'static {
    /// The aggregate state to persist for `input_dtype`, or `None` when unsupported.
    ///
    /// TODO (joacoc): when is it possible to be unsupported?
    fn aggregate_fn(&self, input_dtype: &DType) -> Option<AggregateFnRef>;

    /// Register the aggregate, optional probe function, and predicate rewrite as one operation.
    fn register(&self, session: &VortexSession);
}

// TODO(joacoc): It is more ergonomic, but I still need to check if `SkipIndexSessionExt`
// follows the conventions used by other session extensions.

/// Adds skip-index registration methods to [`VortexSession`].
pub trait SkipIndexSessionExt {
    /// Registers a skip index with this session.
    ///
    /// Register the index with every session that writes or reads files containing it.
    /// Registration makes the index implementation available, but does not add the index to a
    /// file. To write it, also add the [`SkipIndexRef`] to [`ZonedLayoutOptions`].
    ///
    /// # Example
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
    fn register_skip_index(&self, index: &SkipIndexRef);
}

impl SkipIndexSessionExt for VortexSession {
    fn register_skip_index(&self, index: &SkipIndexRef) {
        index.register(self);
    }
}

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
    /// Add `index` to this zoned writer while retaining the default min/max-style aggregates.
    ///
    /// `WriteStrategyBuilder::with_field_zoned_options` can install the configured options for one
    /// field while retaining the default data layout pipeline.
    pub fn with_skip_index(mut self, index: SkipIndexRef) -> VortexResult<Self> {
        let mut skip_indexes: Vec<SkipIndexRef> = self
            .skip_indexes
            .take()
            .map(|arc| arc.to_vec())
            .unwrap_or_default();

        // TODO (joacoc): avoid duplicates
        skip_indexes.push(index);

        self.skip_indexes = Some(skip_indexes.into());
        Ok(self)
    }
}
