/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compute_single_pass_aggs.cuh"
#include "compute_single_pass_aggs.hpp"

namespace cudf::groupby::detail::hash {
template std::pair<rmm::device_uvector<size_type>, bool>
compute_single_pass_aggs<nullable_global_set_t<false>>(nullable_global_set_t<false>& global_set,
                                                       bitmask_type const* row_bitmask,
                                                       host_span<aggregation_request const> requests,
                                                       cudf::detail::result_cache* cache,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::device_async_resource_ref mr);

template std::pair<rmm::device_uvector<size_type>, bool>
compute_single_pass_aggs<nullable_global_set_t<true>>(nullable_global_set_t<true>& global_set,
                                                      bitmask_type const* row_bitmask,
                                                      host_span<aggregation_request const> requests,
                                                      cudf::detail::result_cache* cache,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr);
}  // namespace cudf::groupby::detail::hash
