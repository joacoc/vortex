// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Skipping-index interface and implementations.

use std::fmt::Debug;
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
pub trait SkipIndex: Debug + Send + Sync + 'static {
    /// The aggregate state to persist for `input_dtype`, or `None` when unsupported.
    ///
    /// TODO (joacoc): when is it possible to be unsupported?
    fn aggregate_fn(&self, input_dtype: &DType) -> Option<AggregateFnRef>;

    /// Register the aggregate, optional probe function, and predicate rewrite as one operation.
    fn register(&self, session: &VortexSession);
}

#[derive(Clone)]
pub struct SkipIndexRef(pub(super) Arc<dyn SkipIndex>);

impl ZonedLayoutOptions {
    /// Add `index` to this zoned writer while retaining the default min/max-style aggregates.
    ///
    /// `WriteStrategyBuilder::with_field_zoned_options` can install the configured options for one
    /// field while retaining the default data layout pipeline.
    pub fn with_skip_index<I: SkipIndex>(mut self, index: Arc<I>) -> VortexResult<Self> {
        let mut skip_indexes: Vec<SkipIndexRef> = self
            .skip_indexes
            .take()
            .map(|arc| arc.to_vec())
            .unwrap_or_default();

        // TODO (joacoc): avoid duplicates
        let index: Arc<dyn SkipIndex> = index;
        skip_indexes.push(SkipIndexRef(index));

        self.skip_indexes = Some(skip_indexes.into());
        Ok(self)
    }
}
