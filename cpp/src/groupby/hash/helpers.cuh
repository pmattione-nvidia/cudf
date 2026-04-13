/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/types.hpp>

#include <rmm/mr/polymorphic_allocator.hpp>

#include <cuco/static_set.cuh>

#include <type_traits>

namespace cudf::groupby::detail::hash {
/// Number of threads to handle each input element
CUDF_HOST_DEVICE auto constexpr GROUPBY_CG_SIZE = 1;

/// Number of slots per thread
CUDF_HOST_DEVICE auto constexpr GROUPBY_BUCKET_SIZE = 1;

/// Thread block size
CUDF_HOST_DEVICE auto constexpr GROUPBY_BLOCK_SIZE = 128;

/// Threshold cardinality to switch between shared memory aggregations and global memory
/// aggregations
CUDF_HOST_DEVICE auto constexpr GROUPBY_CARDINALITY_THRESHOLD = 128;

/// Threshold to switch between two strategies: one is to output the aggregation results directly to
/// the final dense output columns, the other is to output the results to sparse intermediate
/// buffers then gather to the final dense output columns.
auto constexpr GROUPBY_DENSE_OUTPUT_THRESHOLD = 2;

// We add additional `block_size`, because after the number of elements in the local hash set
// exceeds the threshold, all threads in the thread block can still insert one more element.
/// The maximum number of elements handled per block
CUDF_HOST_DEVICE auto constexpr GROUPBY_SHM_MAX_ELEMENTS =
  GROUPBY_CARDINALITY_THRESHOLD + GROUPBY_BLOCK_SIZE;

// GROUPBY_SHM_MAX_ELEMENTS with 0.7 occupancy
/// Shared memory hash set extent type
using shmem_extent_t =
  cuco::extent<cudf::size_type,
               static_cast<cudf::size_type>(static_cast<double>(GROUPBY_SHM_MAX_ELEMENTS) * 1.43)>;

/// Number of slots needed by each shared memory hash set
CUDF_HOST_DEVICE auto constexpr valid_extent =
  cuco::make_valid_extent<GROUPBY_CG_SIZE, GROUPBY_BUCKET_SIZE>(shmem_extent_t{});

/// @tparam has_nested Matches `device_row_hasher` / top-level nested key columns (see hash groupby).
template <bool has_nested>
using row_hash_t = cudf::detail::row::hash::device_row_hasher<cudf::hashing::detail::default_hash,
                                                              cudf::nullate::DYNAMIC,
                                                              has_nested>;

/// Adapter to cudf row hasher with caching support.
template <typename Hash>
class row_hasher_with_cache_t {
  Hash hasher;
  hash_value_type const* values;

 public:
  static constexpr bool has_nested = Hash::has_nested_v;

  row_hasher_with_cache_t(Hash const& hasher, hash_value_type const* values = nullptr) noexcept
    : hasher(hasher), values(values)
  {
  }

  __device__ hash_value_type operator()(size_type const idx) const noexcept
  {
    if (values) { return values[idx]; }
    return hasher(idx);
  }
};

template <bool has_nested>
using probing_scheme_t =
  cuco::linear_probing<GROUPBY_CG_SIZE, row_hasher_with_cache_t<row_hash_t<has_nested>>>;

using row_comparator_t = cudf::detail::row::equality::device_row_comparator<
  false,
  cudf::nullate::DYNAMIC,
  cudf::detail::row::equality::nan_equal_physical_equality_comparator>;

using nullable_row_comparator_t = cudf::detail::row::equality::device_row_comparator<
  true,
  cudf::nullate::DYNAMIC,
  cudf::detail::row::equality::nan_equal_physical_equality_comparator>;

template <bool has_nested>
using global_set_t = cuco::static_set<cudf::size_type,
                                      cuco::extent<int64_t>,
                                      cuda::thread_scope_device,
                                      row_comparator_t,
                                      probing_scheme_t<has_nested>,
                                      rmm::mr::polymorphic_allocator<char>,
                                      cuco::storage<GROUPBY_BUCKET_SIZE>>;

template <bool has_nested>
using nullable_global_set_t =
  cuco::static_set<cudf::size_type,
                   cuco::extent<int64_t>,
                   cuda::thread_scope_device,
                   nullable_row_comparator_t,
                   probing_scheme_t<has_nested>,
                   rmm::mr::polymorphic_allocator<char>,
                   cuco::storage<GROUPBY_BUCKET_SIZE>>;

template <bool has_nested, typename Op>
using hash_set_ref_t =
  cuco::static_set_ref<cudf::size_type,
                       cuda::thread_scope_device,
                       row_comparator_t,
                       probing_scheme_t<has_nested>,
                       cuco::bucket_storage_ref<cudf::size_type,
                                                GROUPBY_BUCKET_SIZE,
                                                cuco::valid_extent<int64_t, cuco::dynamic_extent>>,
                       Op>;

template <bool has_nested, typename Op>
using nullable_hash_set_ref_t =
  cuco::static_set_ref<cudf::size_type,
                       cuda::thread_scope_device,
                       nullable_row_comparator_t,
                       probing_scheme_t<has_nested>,
                       cuco::bucket_storage_ref<cudf::size_type,
                                                GROUPBY_BUCKET_SIZE,
                                                cuco::valid_extent<int64_t, cuco::dynamic_extent>>,
                       Op>;

/// `has_nested` for `compute_mapping_indices` / `mapping_indices_kernel` (must match `SetType`'s
/// `probing_scheme_t<has_nested>`).
template <typename SetType>
struct row_hasher_has_nested_for_hash_set;

template <bool H>
struct row_hasher_has_nested_for_hash_set<global_set_t<H>> : std::integral_constant<bool, H> {};

template <bool H>
struct row_hasher_has_nested_for_hash_set<nullable_global_set_t<H>>
  : std::integral_constant<bool, H> {};

template <typename SetType>
inline constexpr bool row_hasher_has_nested_for_hash_set_v =
  row_hasher_has_nested_for_hash_set<SetType>::value;
}  // namespace cudf::groupby::detail::hash
