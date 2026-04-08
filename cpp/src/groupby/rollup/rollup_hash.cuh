/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/hashing.hpp>
#include <cudf/hashing/detail/default_hash.cuh>
#include <cudf/hashing/detail/hashing.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/std/type_traits>

namespace cudf::groupby::detail::hash {

/**
 * @brief Virtual row index layout: `virtual_index = input_row_index * num_levels + grouping_level`.
 *
 * `num_levels` is `rolled_key_count + 1`. `grouping_level` 0 is the finest grouping (all rolled
 * keys participate); `grouping_level == rolled_key_count` is coarsest (only fixed keys
 * participate).
 */
[[nodiscard]] inline size_type rollup_num_levels(size_type num_rolled_keys)
{
  return num_rolled_keys + 1;
}

[[nodiscard]] inline size_type rollup_virtual_row_count(size_type input_rows, size_type num_levels)
{
  return input_rows * num_levels;
}

[[nodiscard]] __device__ inline size_type rollup_virtual_to_input_row(size_type virtual_ix,
                                                                      size_type num_levels)
{
  return virtual_ix / num_levels;
}

[[nodiscard]] __device__ inline size_type rollup_virtual_to_grouping_level(size_type virtual_ix,
                                                                           size_type num_levels)
{
  return virtual_ix % num_levels;
}

/**
 * `rolled_rank[i]` is the position of key column `i` in the rollup suffix (0 .. num_rolled-1), or
 * `num_rolled` when column `i` is a fixed (non-rolled) grouping key.
 */
[[nodiscard]] __device__ inline bool rollup_is_column_active(size_type rolled_rank,
                                                             size_type num_rolled,
                                                             size_type grouping_level)
{
  if (rolled_rank == num_rolled) { return true; }
  return rolled_rank < (num_rolled - grouping_level);
}

/**
 * @brief For `null_policy::EXCLUDE`, skip a virtual row when any *active* key column is null.
 */
[[nodiscard]] __device__ inline bool rollup_skip_virtual_row_for_null_exclude(
  table_device_view table,
  size_type const* rolled_rank,
  size_type num_rolled,
  size_type num_levels,
  size_type virtual_ix)
{
  auto const row            = rollup_virtual_to_input_row(virtual_ix, num_levels);
  auto const grouping_level = rollup_virtual_to_grouping_level(virtual_ix, num_levels);
  for (size_type col = 0; col < table.num_columns(); ++col) {
    if (not rollup_is_column_active(rolled_rank[col], num_rolled, grouping_level)) { continue; }
    if (table.column(col).is_null(row)) { return true; }
  }
  return false;
}

namespace {

using rollup_column_element_hasher_adapter =
  cudf::detail::row::hash::element_hasher_adapter<cudf::hashing::detail::default_hash,
                                                  nullate::DYNAMIC>;

/** Same device element equality as hash `aggregate` (`element_comparator` + `type_dispatcher`). */
using rollup_key_element_equal = cudf::detail::row::equality::element_comparator<
  true,
  nullate::DYNAMIC,
  cudf::detail::row::equality::nan_equal_physical_equality_comparator>;

}  // namespace

/**
 * @brief Probing hash for `cuco::static_set`: 32/64-bit hash of a virtual index.
 *
 * Decodes `virtual_ix` into `(input_row, grouping_level)`, combines `grouping_level` into the hash,
 * then hashes each key column that is *active* at that level (see `rollup_is_column_active`) in
 * column order, using the same element hashing as hash `aggregate` (`element_hasher_adapter` +
 * `type_dispatcher`). Inactive columns are omitted so two virtual rows that differ only on
 * rolled-away keys collide and merge in the set.
 */
struct rollup_row_hasher {
  table_device_view table;
  nullate::DYNAMIC check_nullness;
  size_type num_levels{};
  size_type num_rolled{};
  size_type const* rolled_rank{};

  using result_type =
    cuda::std::invoke_result_t<cudf::hashing::detail::default_hash<int32_t>, int32_t>;

  __device__ result_type operator()(size_type const virtual_ix) const noexcept
  {
    auto const row            = rollup_virtual_to_input_row(virtual_ix, num_levels);
    auto const grouping_level = rollup_virtual_to_grouping_level(virtual_ix, num_levels);

    result_type h = cudf::DEFAULT_HASH_SEED;
    // Level must participate in the hash so the same input row at different rollup levels never
    // collides.
    h = cudf::hashing::detail::hash_combine(h, static_cast<result_type>(grouping_level));

    for (size_type col = 0; col < table.num_columns(); ++col) {
      if (not rollup_is_column_active(rolled_rank[col], num_rolled, grouping_level)) { continue; }

      auto const& column  = table.column(col);
      auto const col_hash = cudf::type_dispatcher<cudf::dispatch_storage_type>(
        column.type(),
        rollup_column_element_hasher_adapter{check_nullness, cudf::DEFAULT_HASH_SEED},
        column,
        row);
      h = cudf::hashing::detail::hash_combine(h, col_hash);
    }
    return h;
  }
};

/**
 * @brief Equality predicate for `cuco::static_set` on virtual indices.
 *
 * Two indices are equal iff they map to the same `grouping_level` and, for every key column active
 * at that level, the underlying input rows have equal elements under the same null / NaN policy as
 * hash `aggregate` (`rollup_key_element_equal`). Inactive columns are ignored, matching
 * `rollup_row_hasher`.
 */
struct rollup_row_equal {
  table_device_view table;
  nullate::DYNAMIC check_nullness;
  null_equality null_keys_are_equal;
  size_type num_levels{};
  size_type num_rolled{};
  size_type const* rolled_rank{};

  __device__ bool operator()(size_type const v1, size_type const v2) const noexcept
  {
    auto const r1     = rollup_virtual_to_input_row(v1, num_levels);
    auto const r2     = rollup_virtual_to_input_row(v2, num_levels);
    auto const level1 = rollup_virtual_to_grouping_level(v1, num_levels);
    auto const level2 = rollup_virtual_to_grouping_level(v2, num_levels);
    if (level1 != level2) { return false; }

    for (size_type col = 0; col < table.num_columns(); ++col) {
      if (not rollup_is_column_active(rolled_rank[col], num_rolled, level1)) { continue; }

      auto const& c = table.column(col);
      // Compare row r1 vs r2 in column c; dispatcher picks physical equality for the column type.
      bool const c_eq = cudf::type_dispatcher(
        c.type(), rollup_key_element_equal{check_nullness, c, c, null_keys_are_equal}, r1, r2);
      if (not c_eq) { return false; }
    }
    return true;
  }
};

}  // namespace cudf::groupby::detail::hash
