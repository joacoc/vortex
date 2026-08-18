//! Aggregate functions selected by the zoned layout.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// (joacoc) Allow dead code here until the writer interface for accessing the
// Bloom filter lands in https://github.com/vortex-data/vortex/pull/9413,
// unless another access point exists that I am unaware of.
#[allow(dead_code)]
pub(in crate::layouts::zoned) mod bloom_filter;
