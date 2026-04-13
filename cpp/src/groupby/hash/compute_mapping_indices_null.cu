/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compute_mapping_indices.cuh"
#include "compute_mapping_indices.hpp"

namespace cudf::groupby::detail::hash {
using nullable_insert_find_ref_false = nullable_hash_set_ref_t<false, cuco::insert_and_find_tag>;
using nullable_insert_find_ref_true  = nullable_hash_set_ref_t<true, cuco::insert_and_find_tag>;

template int32_t max_active_blocks_mapping_kernel<nullable_insert_find_ref_false, false>();
template int32_t max_active_blocks_mapping_kernel<nullable_insert_find_ref_true, true>();

template void compute_mapping_indices<nullable_insert_find_ref_false, false>(
  size_type grid_size,
  size_type num_rows,
  nullable_insert_find_ref_false global_set,
  bitmask_type const* row_bitmask,
  size_type* local_mapping_index,
  size_type* global_mapping_index,
  size_type* block_cardinality,
  cuda::std::atomic_flag* needs_global_memory_fallback,
  rmm::cuda_stream_view stream);

template void compute_mapping_indices<nullable_insert_find_ref_true, true>(
  size_type grid_size,
  size_type num_rows,
  nullable_insert_find_ref_true global_set,
  bitmask_type const* row_bitmask,
  size_type* local_mapping_index,
  size_type* global_mapping_index,
  size_type* block_cardinality,
  cuda::std::atomic_flag* needs_global_memory_fallback,
  rmm::cuda_stream_view stream);
}  // namespace cudf::groupby::detail::hash
