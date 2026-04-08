/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// Hash ROLLUP: map each (input row, grouping level) to a virtual index, dedupe with a device
// hash set keyed by active key columns + level, aggregate into a sparse table indexed by virtual
// index, compact to dense rows, materialize output keys (nulls where keys are rolled away), append
// group_id, then reuse hash aggregate finalize / compound-agg handling.

#include "rollup_aggregate_kernel.cuh"

#include "../common/utils.hpp"
#include "../hash/extract_single_pass_aggs.hpp"
#include "../hash/hash_compound_agg_finalizer.hpp"
#include "../hash/output_utils.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/aggregation/result_cache.hpp>
#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/groupby.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/row_operator/preprocessed_table.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/groupby.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <cuco/static_set.cuh>

#include <cuda_runtime.h>

#include <rmm/exec_policy.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace cudf::groupby::detail::hash {

/**
 * @brief For each output key row (one per unique virtual index), mark BOOL8 valid where the key
 *        column is active for that row's grouping level; leave null where the column is rolled away.
 */
CUDF_KERNEL void rollup_active_key_column_kernel(mutable_column_device_view active_bool,
                                                 size_type const* unique_virtual,
                                                 size_type num_unique,
                                                 size_type rolled_rank_c,
                                                 size_type num_rolled,
                                                 size_type num_levels)
{
  // One thread per output row; out-of-range threads exit (grid sized to num_unique).
  auto const i = static_cast<size_type>(blockDim.x * blockIdx.x + threadIdx.x);
  if (i >= num_unique) { return; }

  // Decode virtual index to grouping level; BOOL8 stays null if this key column is rolled off at that level.
  auto const v     = unique_virtual[i];
  auto const level = v % num_levels;
  if (rollup_is_column_active(rolled_rank_c, num_rolled, level)) {
    active_bool.set_valid(i);
    active_bool.element<uint8_t>(i) = 1;
  }
}

}  // namespace cudf::groupby::detail::hash

namespace {

using cudf::size_type;

// Row bitmask for EXCLUDE_NULL_KEYS when keys have multiple columns (single-column fast path above).
[[nodiscard]] std::pair<rmm::device_buffer, cudf::bitmask_type const*> rollup_keys_row_bitmask(
  cudf::table_view const& keys,
  bool exclude_null_keys,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  // Compound-agg finalizer uses nullptr to mean "no per-row key null filter" (include null keys or no nulls).
  if (not exclude_null_keys or not cudf::has_nulls(keys)) {
    return {rmm::device_buffer{}, nullptr};
  }

  // One key column: row is excluded iff that column is null; reuse mask or align to offset 0 via copy.
  if (keys.num_columns() == 1) {
    auto const& keys_col = keys.column(0);
    if (keys_col.offset() == 0) { return {rmm::device_buffer{}, keys_col.null_mask()}; }
    auto null_mask_data  = cudf::copy_bitmask(keys_col, stream);
    auto const null_mask = static_cast<cudf::bitmask_type const*>(null_mask_data.data());
    return {std::move(null_mask_data), null_mask};
  }

  // Multiple keys: a row is in the group iff no key column is null at that row (bitwise AND of null masks).
  auto [null_mask_data, null_count] = cudf::bitmask_and(keys, stream, mr);
  if (null_count == 0) { return {rmm::device_buffer{}, nullptr}; }
  auto const null_mask = static_cast<cudf::bitmask_type const*>(null_mask_data.data());
  return {std::move(null_mask_data), null_mask};
}

// One output row per unique virtual index: gather each key column from the input row (v / num_levels),
// then AND with an active-column BOOL mask so rolled-away keys become null.
[[nodiscard]] std::unique_ptr<cudf::table> materialize_rollup_output_keys(
  cudf::table_view const& keys_table,
  rmm::device_uvector<size_type> const& d_unique_virtual,
  size_type num_levels,
  size_type num_rolled,
  std::vector<size_type> const& h_rolled_rank,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const num_unique = static_cast<size_type>(d_unique_virtual.size());
  if (num_unique == 0) { return cudf::empty_like(keys_table); }

  // Map each canonical virtual index to its source input row: virtual = row * num_levels + level, so row = v / num_levels.
  rmm::device_uvector<size_type> d_input_row(num_unique, stream);
  thrust::transform(
    rmm::exec_policy_nosync(stream),
    d_unique_virtual.begin(),
    d_unique_virtual.end(),
    d_input_row.begin(),
    [num_levels] __device__(size_type const v) { return v / num_levels; });

  // Gather uses these row ids in order; indices are in-range because virtual indices are derived from input rows.
  auto const gather_span = cudf::device_span<size_type const>{d_input_row.data(), d_input_row.size()};

  std::vector<std::unique_ptr<cudf::column>> out_cols;
  out_cols.reserve(static_cast<std::size_t>(keys_table.num_columns()));

  constexpr int key_k_threads = 256;
  int const key_k_grid =
    static_cast<int>(cudf::util::div_rounding_up_safe(num_unique, key_k_threads));

  for (size_type c = 0; c < keys_table.num_columns(); ++c) {
    // Pull key values from the original table for each output row (one row per unique virtual index).
    auto gathered_tbl = cudf::detail::gather(
      cudf::table_view{{keys_table.column(c)}},
      gather_span,
      cudf::out_of_bounds_policy::DONT_CHECK,
      cudf::negative_index_policy::NOT_ALLOWED,
      stream,
      mr);

    auto released = gathered_tbl->release();
    auto gathered = std::move(released.front());

    // BOOL8 marks where this column participates at each row's grouping level; starts all-null, kernel sets valid+1 where active.
    auto active = cudf::make_numeric_column(cudf::data_type{cudf::type_id::BOOL8},
                                            num_unique,
                                            cudf::mask_state::ALL_NULL,
                                            stream,
                                            mr);

    auto d_active        = cudf::mutable_column_device_view::create(active->mutable_view(), stream);
    auto const rolled_rc = h_rolled_rank[static_cast<std::size_t>(c)];
    cudf::groupby::detail::hash::rollup_active_key_column_kernel<<<key_k_grid, key_k_threads, 0, stream.value()>>>(
      *d_active,
      d_unique_virtual.data(),
      num_unique,
      rolled_rc,
      num_rolled,
      num_levels);
    CUDF_CUDA_TRY(cudaPeekAtLastError());

    // Output key is null where the column is inactive (rolled) or where the gathered value was null.
    auto const [merged_mask, null_count] =
      cudf::bitmask_and(cudf::table_view{{gathered->view(), active->view()}}, stream, mr);
    gathered->set_null_mask(std::move(merged_mask), null_count);
    out_cols.push_back(std::move(gathered));
  }

  return std::make_unique<cudf::table>(std::move(out_cols));
}

// INT64 per output row: `(1 << grouping_level) - 1` encodes which rolled keys are active at that level.
[[nodiscard]] std::unique_ptr<cudf::column> make_group_id_column(
  rmm::device_uvector<size_type> const& unique_virtual,
  size_type num_levels,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(num_levels > 0 && num_levels < 63,
               "rollup: num_levels out of range for group_id");
  auto const n = static_cast<size_type>(unique_virtual.size());
  if (n == 0) { return cudf::make_empty_column(cudf::type_id::INT64); }

  // No nulls: group_id is a pure function of grouping level; consumers use it to tell rollup levels apart.
  auto out = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT64}, n, cudf::mask_state::UNALLOCATED, stream, mr);
  auto* d_out = out->mutable_view().data<int64_t>();

  // Per output row, level = v % num_levels; store a small bitmask (1<<level)-1 identifying which rolled dims exist at that level.
  thrust::transform(
    rmm::exec_policy_nosync(stream),
    unique_virtual.begin(),
    unique_virtual.end(),
    d_out,
    [num_levels] __device__(size_type const v) -> int64_t {
      auto const g = v % num_levels;
      return static_cast<int64_t>((static_cast<std::uint64_t>(1) << g) - 1);
    });
  return out;
}

// Insert group_id immediately after the key columns (before aggregation result columns in the API).
[[nodiscard]] std::unique_ptr<cudf::table> append_column_after_keys(
  std::unique_ptr<cudf::table> keys_table,
  std::unique_ptr<cudf::column> extra,
  size_type num_key_columns)
{
  // Take ownership of key columns as a vector so we can append without copying column data.
  auto cols = keys_table->release();
  CUDF_EXPECTS(static_cast<size_type>(cols.size()) == num_key_columns,
               "rollup: key table column count mismatch");
  cols.push_back(std::move(extra));
  return std::make_unique<cudf::table>(std::move(cols));
}

}  // namespace

namespace cudf::groupby::detail::hash {

// Used to size the insert_find kernel grid (one block per SM, each block covers a slice of rows).
[[nodiscard]] inline size_type device_multiprocessor_count()
{
  int device = 0;
  CUDF_CUDA_TRY(cudaGetDevice(&device));
  int num_sms = 0;
  CUDF_CUDA_TRY(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));
  return static_cast<size_type>(num_sms);
}

std::pair<std::unique_ptr<table>, std::vector<aggregation_result>> rollup(
  table_view const& keys,
  host_span<aggregation_request const> requests,
  host_span<size_type const> rolled_up_key_column_indices,
  null_policy include_null_keys,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  // Hash rollup shares kernels and storage layouts with single-pass hash aggregate; nested keys and unsupported aggs bail early.
  CUDF_EXPECTS(not cudf::has_nested_columns(keys),
               "rollup: nested key columns are not supported in the hash path");
  CUDF_EXPECTS(not requests.empty(), "rollup aggregate requires at least one aggregation request");
  CUDF_EXPECTS(cudf::groupby::detail::hash::can_use_hash_groupby(requests),
               "rollup: one or more aggregations are not supported on the hash path");

  // Degenerate case: same as hash aggregate with an empty group_id column (no virtual expansion).
  if (keys.num_rows() == 0) {
    cudf::scoped_range const empty_range{"detail::hash::rollup: empty keys"};
    cudf::groupby::groupby gb(keys, include_null_keys, sorted::NO, {}, {});

    // No groups to form; still produce the same result schema as non-empty rollup (keys + group_id + agg columns).
    auto empty_out = gb.aggregate(requests, stream, mr);
    auto group_id  = cudf::make_empty_column(cudf::type_id::INT64);
    empty_out.first =
      append_column_after_keys(std::move(empty_out.first), std::move(group_id), keys.num_columns());
    return empty_out;
  }

  // Preprocess keys for row equality / hashing (same representation as hash groupby).
  auto const preprocessed_keys =
    cudf::detail::row::equality::preprocessed_table::create(keys, stream);
  auto const keys_dview = static_cast<table_device_view>(*preprocessed_keys);

  // Virtual index space: each input row spawns num_levels rows (one per rollup granularity); total slots = rows * levels.
  auto const num_input_rows = static_cast<size_type>(keys.num_rows());
  auto const num_rolled     = static_cast<size_type>(rolled_up_key_column_indices.size());
  auto const num_levels     = rollup_num_levels(num_rolled);
  auto const num_virtual    = rollup_virtual_row_count(num_input_rows, num_levels);
  auto const key_nullate    = cudf::nullate::DYNAMIC{cudf::has_nulls(keys)};

  // Host-side rank: for each key column, position in the rollup suffix or num_rolled if fixed key.
  std::vector<size_type> h_rank(static_cast<std::size_t>(keys.num_columns()), num_rolled);
  for (std::size_t p = 0; p < rolled_up_key_column_indices.size(); ++p) {
    h_rank[static_cast<std::size_t>(rolled_up_key_column_indices[p])] = static_cast<size_type>(p);
  }

  // Device copy so kernels can test column activity without host access (rolled_rank[col] vs grouping level).
  auto d_rolled_rank = cudf::detail::make_device_uvector_async(h_rank, stream, mr);

  // Flatten requests into one values table, aggregation kinds, and host aggs for finalize (same as hash aggregate).
  auto const [values, agg_kinds, aggs, is_agg_intermediate, has_compound_aggs] =
    cudf::groupby::detail::hash::extract_single_pass_aggs(requests, stream);
  CUDF_EXPECTS(values.num_rows() == static_cast<size_type>(keys.num_rows()),
               "rollup: values row count must match keys row count");

  // Sparse aggregate storage: one logical slot per virtual row (input_row * num_levels + level).
  auto sparse_agg_results = cudf::groupby::detail::hash::create_results_table(
    num_virtual,
    values,
    agg_kinds,
    std::span<std::int8_t const>{is_agg_intermediate.data(), is_agg_intermediate.size()},
    stream,
    mr);

  // Device views: read value rows from the input table, read/write aggregate partials in the sparse results table.
  auto d_input_values  = table_device_view::create(values, stream);
  auto d_output_values = mutable_table_device_view::create(sparse_agg_results->mutable_view(), stream);
  auto d_agg_kinds     = cudf::detail::make_device_uvector_async(agg_kinds, stream, mr);

  // Functors encode virtual-index equality and hash using only keys active at that virtual row's level (see rollup_hash.cuh).
  rollup_row_equal row_equal{
    keys_dview, key_nullate, null_equality::EQUAL, num_levels, num_rolled, d_rolled_rank.data()};
  rollup_row_hasher row_hash{keys_dview, key_nullate, num_levels, num_rolled, d_rolled_rank.data()};

  // cuco open addressing: rehash on collision using the same rollup_row_hasher step (stride 1).
  using probing_scheme_t = cuco::linear_probing<1, rollup_row_hasher>;
  probing_scheme_t probing_scheme{row_hash};

  // Device-wide set: payload is the winning virtual index for each distinct (active keys, level) equivalence class.
  using rollup_global_set_t = cuco::static_set<size_type,
                                               cuco::extent<int64_t>,
                                               cuda::thread_scope_device,
                                               rollup_row_equal,
                                               probing_scheme_t,
                                               rmm::mr::polymorphic_allocator<char>,
                                               cuco::storage<1>>;

  // Set stores canonical virtual indices (size_type); capacity is upper bound count of distinct groups (<= num_virtual).
  auto set = rollup_global_set_t{cuco::extent<int64_t>{static_cast<int64_t>(num_virtual)},
                                 cudf::detail::CUCO_DESIRED_LOAD_FACTOR,
                                 cuco::empty_key{cudf::detail::CUDF_SIZE_TYPE_SENTINEL},
                                 row_equal,
                                 probing_scheme,
                                 cuco::thread_scope_device,
                                 cuco::storage<1>{},
                                 rmm::mr::polymorphic_allocator<char>{},
                                 stream.value()};

  // EXCLUDE matches hash groupby: skip virtual rows whose active key slice contains a null.
  bool const exclude_null_keys = include_null_keys == null_policy::EXCLUDE;

  // insert_and_find ref: map each probe to a slot and merge aggregates when keys match an existing virtual index.
  auto set_ref_insert_find = set.ref(cuco::op::insert_and_find);
  constexpr int block_size   = 256;
  size_type const num_sms      = std::max(device_multiprocessor_count(), size_type{1});
  size_type const rows_per_block =
    num_input_rows > 0 ? cudf::util::div_rounding_up_safe(num_input_rows, num_sms) : 0;
  int const grid_size = static_cast<int>(num_sms);
  if (num_input_rows > 0) {
    cudf::scoped_range const kernel_range{"detail::hash::rollup: insert_find kernel"};
    // One block per SM; each thread walks a chunk of input rows and all levels, hashing into the set and updating sparse aggs.
    rollup_aggregate_insert_find_kernel<<<grid_size, block_size, 0, stream.value()>>>(
      num_input_rows,
      rows_per_block,
      num_levels,
      exclude_null_keys,
      keys_dview,
      d_rolled_rank.data(),
      num_rolled,
      *d_input_values,
      *d_output_values,
      d_agg_kinds.data(),
      set_ref_insert_find);
    CUDF_CUDA_TRY(cudaPeekAtLastError());
  }

  // Buffer sized to worst case (every virtual slot occupied); retrieve_all writes unique keys then we shrink to actual count.
  rmm::device_uvector<size_type> unique_virtual_indices(static_cast<std::size_t>(num_virtual), stream, mr);
  {
    cudf::scoped_range const retrieve_range{"detail::hash::rollup: retrieve unique virtual indices"};
    auto const keys_end = set.retrieve_all(unique_virtual_indices.begin(), stream.value());
    unique_virtual_indices.resize(
      static_cast<std::size_t>(std::distance(unique_virtual_indices.begin(), keys_end)), stream);
  }

  std::unique_ptr<table> dense_agg_table;
  std::unique_ptr<table> unique_keys_table;
  {
    cudf::scoped_range const densify_range{"detail::hash::rollup: densify keys and aggregates"};
    // Compact sparse aggregate rows: keep only slots that correspond to distinct groups (order matches retrieve_all).
    dense_agg_table = cudf::detail::gather(
      sparse_agg_results->view(),
      cudf::device_span<size_type const>{unique_virtual_indices.data(), unique_virtual_indices.size()},
      cudf::out_of_bounds_policy::DONT_CHECK,
      cudf::negative_index_policy::NOT_ALLOWED,
      stream,
      mr);

    // Build key columns aligned to dense output rows; null out rolled-away dimensions per grouping level.
    unique_keys_table =
      materialize_rollup_output_keys(keys, unique_virtual_indices, num_levels, num_rolled, h_rank, stream, mr);

    // Append metadata column so callers can recover rollup level without re-deriving from key nulls alone.
    auto group_id_col = make_group_id_column(unique_virtual_indices, num_levels, stream, mr);
    unique_keys_table =
      append_column_after_keys(std::move(unique_keys_table), std::move(group_id_col), keys.num_columns());
  }

  std::vector<aggregation_result> agg_results;
  {
    cudf::scoped_range const finalize_range{"detail::hash::rollup: finalize aggregations"};
    // Populate cache from dense single-pass columns (null counts, typed outputs) using the same code path as hash aggregate.
    cudf::detail::result_cache cache(requests.size());
    cudf::groupby::detail::hash::finalize_output(values, aggs, dense_agg_table, &cache, stream);

    // Compound aggs may need to drop rows where original keys were null; bitmask mirrors hash aggregate behavior.
    auto [row_bitmask_storage, row_bitmask] =
      rollup_keys_row_bitmask(keys, exclude_null_keys, stream, mr);

    if (has_compound_aggs) {
      // Dispatch each compound aggregation kind to a finalizer that reads cached intermediates (same as hash groupby).
      for (auto const& request : requests) {
        auto const finalizer = cudf::groupby::detail::hash::hash_compound_agg_finalizer(
          request.values, &cache, row_bitmask, stream, mr);
        for (auto const& agg : request.aggregations) {
          cudf::detail::aggregation_dispatcher(agg->kind, finalizer, *agg);
        }
      }
    }

    // One aggregation_result per request column, matching public groupby::rollup contract.
    agg_results = cudf::groupby::detail::extract_results(requests, cache, stream, mr);
  }
  return {std::move(unique_keys_table), std::move(agg_results)};
}

}  // namespace cudf::groupby::detail::hash
