/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "rollup_hash.cuh"

#include <cudf/aggregation.hpp>
#include <cudf/detail/aggregation/device_aggregators.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>

namespace cudf::groupby::detail::hash {

/**
 * @brief Grid is one block per SM; each block covers `rows_per_block` consecutive input rows.
 *
 * Threads stride over that row range, then the inner loop runs rollup grouping levels (group id).
 * For each virtual key: `insert_and_find` (sparse slot = canonical virtual index), then for each
 * value column apply the same `dispatch_type_and_aggregation` path as hash single-pass groupby.
 *
 * Virtual index: `virtual_ix = input_row * num_levels + group_id`.
 */
template <typename SetRef>
CUDF_KERNEL void rollup_aggregate_insert_find_kernel(size_type num_input_rows,
                                                     size_type rows_per_block,
                                                     size_type num_levels,
                                                     bool exclude_null_keys,
                                                     table_device_view keys_dview,
                                                     size_type const* rolled_rank,
                                                     size_type num_rolled,
                                                     table_device_view input_values,
                                                     mutable_table_device_view output_values,
                                                     aggregation::Kind const* d_agg_kinds,
                                                     SetRef set_ref)
{
  auto const row_base = static_cast<size_type>(blockIdx.x) * rows_per_block;

  for (size_type local = threadIdx.x; local < rows_per_block; local += blockDim.x) {
    size_type const row = row_base + local;
    if (row >= num_input_rows) { continue; }

    for (size_type group_id = 0; group_id < num_levels; ++group_id) {
      size_type const virtual_ix = row * num_levels + group_id;
      if (exclude_null_keys && rollup_skip_virtual_row_for_null_exclude(
                                 keys_dview, rolled_rank, num_rolled, num_levels, virtual_ix)) {
        continue;
      }

      auto const target_row_idx = *set_ref.insert_and_find(virtual_ix).first;

      for (size_type col_idx = 0; col_idx < input_values.num_columns(); ++col_idx) {
        auto const& source_col = input_values.column(col_idx);
        auto& target_col       = output_values.column(col_idx);
        cudf::detail::dispatch_type_and_aggregation(source_col.type(),
                                                    d_agg_kinds[col_idx],
                                                    cudf::detail::element_aggregator{},
                                                    target_col,
                                                    target_row_idx,
                                                    source_col,
                                                    row);
      }
    }
  }
}

}  // namespace cudf::groupby::detail::hash
