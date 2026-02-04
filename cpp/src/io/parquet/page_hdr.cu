/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "error.hpp"
#include "io/utilities/block_utils.cuh"
#include "page_hdr.cuh"
#include "parquet_gpu.hpp"

#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cooperative_groups.h>
#include <cuda/std/tuple>
#include <thrust/binary_search.h>

#include <vector>

namespace cudf::io::parquet::detail {

namespace {

auto constexpr decode_page_headers_block_size     = 4 * cudf::detail::warp_size;
auto constexpr count_page_headers_block_size      = 4 * cudf::detail::warp_size;
auto constexpr build_string_dict_index_block_size = 4 * cudf::detail::warp_size;
auto constexpr cpu_decode_threshold               = 8;

namespace cg = cooperative_groups;

/**
 * @brief CPU implementation to count page headers for a single chunk
 *
 * @param[in,out] chunk Column chunk to process
 * @param[out] error Pointer to error code
 */
void count_page_headers_chunk_cpu(ColumnChunkDesc& chunk, kernel_error::value_type* error)
{
  byte_stream_s bs{};
  bs.ck   = chunk;
  bs.base = bs.cur = chunk.compressed_data;
  bs.end           = bs.base + chunk.compressed_size;

  size_t const num_values        = chunk.num_values;
  size_t values_found            = 0;
  uint32_t data_page_count       = 0;
  uint32_t dictionary_page_count = 0;

  while (values_found < num_values && bs.cur < bs.end) {
    if (parse_page_header_fn{}(&bs) && bs.page.compressed_page_size >= 0) {
      if (not is_supported_encoding(bs.page.encoding)) {
        *error |= static_cast<kernel_error::value_type>(decode_error::UNSUPPORTED_ENCODING);
      }
      switch (bs.page_type) {
        case PageType::DATA_PAGE:
          data_page_count++;
          values_found += bs.page.num_input_values;
          break;
        case PageType::DATA_PAGE_V2:
          data_page_count++;
          values_found += bs.page.num_input_values;
          break;
        case PageType::DICTIONARY_PAGE: dictionary_page_count++; break;
        default:
          *error |= static_cast<kernel_error::value_type>(decode_error::INVALID_PAGE_TYPE);
          bs.cur = bs.end;
          break;
      }
      bs.cur += bs.page.compressed_page_size;
      if (bs.cur > bs.end) {
        *error |= static_cast<kernel_error::value_type>(decode_error::DATA_STREAM_OVERRUN);
      }
    } else {
      *error |= static_cast<kernel_error::value_type>(decode_error::INVALID_PAGE_HEADER);
      bs.cur = bs.end;
    }
  }

  chunk.num_data_pages = data_page_count;
  chunk.num_dict_pages = dictionary_page_count;
}

/**
 * @brief CPU implementation to decode page headers for a single chunk
 *
 * @param[in] chunk Column chunk to process
 * @param[out] page_info Output array for page information
 * @param[in] chunk_idx Index of this chunk
 * @param[out] error Pointer to error code
 */
void decode_page_headers_chunk_cpu(ColumnChunkDesc const& chunk,
                                   PageInfo* page_info,
                                   cudf::size_type chunk_idx,
                                   kernel_error::value_type* error,
                                   uint8_t const* device_compressed_data_base)
{
  byte_stream_s bs{};
  bs.ck   = chunk;
  bs.base = bs.cur       = chunk.compressed_data;
  bs.end                 = bs.base + chunk.compressed_size;
  bs.page.chunk_idx      = chunk_idx;
  bs.page.src_col_schema = chunk.src_col_schema;
  zero_out_page_header_info(&bs);

  size_t const num_values        = chunk.num_values;
  size_t values_found            = 0;
  uint32_t data_page_count       = 0;
  uint32_t dictionary_page_count = 0;
  auto const max_num_pages       = chunk.num_data_pages + chunk.num_dict_pages;
  auto const num_dict_pages      = chunk.num_dict_pages;

  while (values_found < num_values && bs.cur < bs.end) {
    int index_out = -1;

    // this computation is only valid for flat schemas. for nested schemas,
    // they will be recomputed in the preprocess step by examining repetition and
    // definition levels
    bs.page.chunk_row += bs.page.num_rows;
    bs.page.num_rows      = 0;
    bs.page.flags         = 0;
    bs.page.str_bytes     = 0;
    bs.page.str_bytes_all = 0;
    // zero out V2 info
    bs.page.num_nulls                         = 0;
    bs.page.lvl_bytes[level_type::DEFINITION] = 0;
    bs.page.lvl_bytes[level_type::REPETITION] = 0;

    if (parse_page_header_fn{}(&bs) && bs.page.compressed_page_size >= 0) {
      if (not is_supported_encoding(bs.page.encoding)) {
        *error |= static_cast<kernel_error::value_type>(decode_error::UNSUPPORTED_ENCODING);
      }
      switch (bs.page_type) {
        case PageType::DATA_PAGE:
          index_out = num_dict_pages + data_page_count;
          data_page_count++;
          // this computation is only valid for flat schemas. for nested schemas,
          // they will be recomputed in the preprocess step by examining repetition and
          // definition levels
          bs.page.num_rows = bs.page.num_input_values;
          values_found += bs.page.num_input_values;
          break;
        case PageType::DATA_PAGE_V2:
          index_out = num_dict_pages + data_page_count;
          data_page_count++;
          bs.page.flags |= PAGEINFO_FLAGS_V2;
          values_found += bs.page.num_input_values;
          // V2 only uses RLE, so it was removed from the header
          bs.page.definition_level_encoding = Encoding::RLE;
          bs.page.repetition_level_encoding = Encoding::RLE;
          break;
        case PageType::DICTIONARY_PAGE:
          index_out = dictionary_page_count;
          dictionary_page_count++;
          bs.page.flags |= PAGEINFO_FLAGS_DICTIONARY;
          break;
        default:
          *error |= static_cast<kernel_error::value_type>(decode_error::INVALID_PAGE_TYPE);
          bs.cur = bs.end;
          break;
      }
      // Set page_data pointer - need to translate from host pointer to device pointer
      auto const host_offset = bs.cur - bs.base;
      bs.page.page_data      = const_cast<uint8_t*>(device_compressed_data_base + host_offset);
      bs.cur += bs.page.compressed_page_size;
      if (bs.cur > bs.end) {
        *error |= static_cast<kernel_error::value_type>(decode_error::DATA_STREAM_OVERRUN);
      }
      bs.page.kernel_mask = kernel_mask_for_page(bs.page, bs.ck);
    } else {
      *error |= static_cast<kernel_error::value_type>(decode_error::INVALID_PAGE_HEADER);
      bs.cur = bs.end;
    }
    if (index_out >= 0 && index_out < max_num_pages) { page_info[index_out] = bs.page; }
  }
}

/**
 * @brief Kernel for outputting page headers from the specified column chunks
 *
 * @param[in] chunks Device span of column chunks
 * @param[out] chunk_pages List of chunk-sorted page info (headers)
 * @param[out] error_code Pointer to the error code for kernel failures
 */
CUDF_KERNEL
void __launch_bounds__(decode_page_headers_block_size)
  decode_page_headers_kernel(device_span<ColumnChunkDesc const> chunks,
                             chunk_page_info* chunk_pages,
                             kernel_error::pointer error_code)
{
  auto constexpr num_warps_per_block = decode_page_headers_block_size / cudf::detail::warp_size;

  auto const block = cg::this_thread_block();
  auto const warp  = cg::tiled_partition<cudf::detail::warp_size>(block);

  auto const lane_id = warp.thread_rank();
  auto const warp_id = warp.meta_group_rank();
  auto const chunk_idx =
    static_cast<cudf::size_type>((cg::this_grid().block_rank() * num_warps_per_block) + warp_id);
  auto const num_chunks = static_cast<cudf::size_type>(chunks.size());

  __shared__ byte_stream_s bs_g[num_warps_per_block];
  __shared__ kernel_error::value_type error[num_warps_per_block];

  auto const bs = &bs_g[warp_id];

  if (lane_id == 0) {
    if (chunk_idx < num_chunks) { bs->ck = chunks[chunk_idx]; }
    error[warp_id] = 0;
  }
  block.sync();

  if (chunk_idx < num_chunks) {
    if (lane_id == 0) {
      bs->base = bs->cur      = bs->ck.compressed_data;
      bs->end                 = bs->base + bs->ck.compressed_size;
      bs->page.chunk_idx      = chunk_idx;
      bs->page.src_col_schema = bs->ck.src_col_schema;
      zero_out_page_header_info(bs);
    }
    size_t const num_values        = bs->ck.num_values;
    size_t values_found            = 0;
    uint32_t data_page_count       = 0;
    uint32_t dictionary_page_count = 0;
    auto* page_info                = chunk_pages[chunk_idx].pages;
    auto const max_num_pages       = bs->ck.num_data_pages + bs->ck.num_dict_pages;
    auto const num_dict_pages      = bs->ck.num_dict_pages;
    warp.sync();

    while (values_found < num_values and bs->cur < bs->end) {
      int index_out = -1;

      if (lane_id == 0) {
        // this computation is only valid for flat schemas. for nested schemas,
        // they will be recomputed in the preprocess step by examining repetition and
        // definition levels
        bs->page.chunk_row += bs->page.num_rows;
        bs->page.num_rows      = 0;
        bs->page.flags         = 0;
        bs->page.str_bytes     = 0;
        bs->page.str_bytes_all = 0;
        // zero out V2 info
        bs->page.num_nulls                         = 0;
        bs->page.lvl_bytes[level_type::DEFINITION] = 0;
        bs->page.lvl_bytes[level_type::REPETITION] = 0;
        if (parse_page_header_fn{}(bs) and bs->page.compressed_page_size >= 0) {
          if (not is_supported_encoding(bs->page.encoding)) {
            error[warp_id] |=
              static_cast<kernel_error::value_type>(decode_error::UNSUPPORTED_ENCODING);
          }
          switch (bs->page_type) {
            case PageType::DATA_PAGE:
              index_out = num_dict_pages + data_page_count;
              data_page_count++;
              // this computation is only valid for flat schemas. for nested schemas,
              // they will be recomputed in the preprocess step by examining repetition and
              // definition levels
              bs->page.num_rows = bs->page.num_input_values;
              values_found += bs->page.num_input_values;
              break;
            case PageType::DATA_PAGE_V2:
              index_out = num_dict_pages + data_page_count;
              data_page_count++;
              bs->page.flags |= PAGEINFO_FLAGS_V2;
              values_found += bs->page.num_input_values;
              // V2 only uses RLE, so it was removed from the header
              bs->page.definition_level_encoding = Encoding::RLE;
              bs->page.repetition_level_encoding = Encoding::RLE;
              break;
            case PageType::DICTIONARY_PAGE:
              index_out = dictionary_page_count;
              dictionary_page_count++;
              bs->page.flags |= PAGEINFO_FLAGS_DICTIONARY;
              break;
            default:
              error[warp_id] |=
                static_cast<kernel_error::value_type>(decode_error::INVALID_PAGE_TYPE);
              bs->cur = bs->end;
              break;
          }
          bs->page.page_data = const_cast<uint8_t*>(bs->cur);
          bs->cur += bs->page.compressed_page_size;
          if (bs->cur > bs->end) {
            error[warp_id] |=
              static_cast<kernel_error::value_type>(decode_error::DATA_STREAM_OVERRUN);
          }
          bs->page.kernel_mask = kernel_mask_for_page(bs->page, bs->ck);
        } else {
          error[warp_id] |=
            static_cast<kernel_error::value_type>(decode_error::INVALID_PAGE_HEADER);
          bs->cur = bs->end;
        }
        if (index_out >= 0 and index_out < max_num_pages) { page_info[index_out] = bs->page; }
      }
      values_found = shuffle(values_found);
      warp.sync();
    }
    if (lane_id == 0 and error[warp_id] != 0) { set_error(error[warp_id], error_code); }
  }
}

/**
 * @brief Kernel for counting the number of page headers from the specified column chunks
 *
 * @param[in] chunks Device span of column chunks
 * @param[out] error_code Pointer to the error code for kernel failures
 */
CUDF_KERNEL void __launch_bounds__(count_page_headers_block_size)
  count_page_headers_kernel(cudf::device_span<ColumnChunkDesc> chunks,
                            kernel_error::pointer error_code)
{
  auto constexpr num_warps_per_block = decode_page_headers_block_size / cudf::detail::warp_size;

  auto const block = cg::this_thread_block();
  auto const warp  = cg::tiled_partition<cudf::detail::warp_size>(block);

  auto const lane_id = warp.thread_rank();
  auto const warp_id = warp.meta_group_rank();
  auto const chunk_idx =
    static_cast<cudf::size_type>((cg::this_grid().block_rank() * num_warps_per_block) + warp_id);
  auto const num_chunks = static_cast<cudf::size_type>(chunks.size());

  __shared__ byte_stream_s bs_g[num_warps_per_block];
  __shared__ kernel_error::value_type error[num_warps_per_block];

  auto const bs = &bs_g[warp_id];

  if (lane_id == 0) {
    if (chunk_idx < num_chunks) { bs->ck = chunks[chunk_idx]; }
    error[warp_id] = 0;
  }
  block.sync();

  if (chunk_idx < num_chunks) {
    if (lane_id == 0) {
      bs->base = bs->cur = bs->ck.compressed_data;
      bs->end            = bs->base + bs->ck.compressed_size;
    }
    size_t const num_values        = bs->ck.num_values;
    size_t values_found            = 0;
    uint32_t data_page_count       = 0;
    uint32_t dictionary_page_count = 0;
    warp.sync();
    while (values_found < num_values and bs->cur < bs->end) {
      if (lane_id == 0) {
        if (parse_page_header_fn{}(bs) and bs->page.compressed_page_size >= 0) {
          if (not is_supported_encoding(bs->page.encoding)) {
            error[warp_id] |=
              static_cast<kernel_error::value_type>(decode_error::UNSUPPORTED_ENCODING);
          }
          switch (bs->page_type) {
            case PageType::DATA_PAGE:
              data_page_count++;
              values_found += bs->page.num_input_values;
              break;
            case PageType::DATA_PAGE_V2:
              data_page_count++;
              values_found += bs->page.num_input_values;
              break;
            case PageType::DICTIONARY_PAGE: dictionary_page_count++; break;
            default:
              error[warp_id] |=
                static_cast<kernel_error::value_type>(decode_error::INVALID_PAGE_TYPE);
              bs->cur = bs->end;
              break;
          }
          bs->cur += bs->page.compressed_page_size;
          if (bs->cur > bs->end) {
            error[warp_id] |=
              static_cast<kernel_error::value_type>(decode_error::DATA_STREAM_OVERRUN);
          }
        } else {
          error[warp_id] |=
            static_cast<kernel_error::value_type>(decode_error::INVALID_PAGE_HEADER);
          bs->cur = bs->end;
        }
      }
      values_found = shuffle(values_found);
      warp.sync();
    }
    if (lane_id == 0) {
      chunks[chunk_idx].num_data_pages = data_page_count;
      chunks[chunk_idx].num_dict_pages = dictionary_page_count;
      if (error[warp_id] != 0) { set_error(error[warp_id], error_code); }
    }
  }
}

/**
 * @brief Functor to decode page headers from specified page locations
 */
struct decode_page_headers_with_pgidx_fn {
  cudf::device_span<ColumnChunkDesc const> colchunks;
  cudf::device_span<PageInfo> pages;
  uint8_t** page_locations;
  size_type* chunk_page_offsets;
  kernel_error::pointer error_code;

  __device__ void operator()(size_type page_idx) const noexcept
  {
    auto const num_chunks = static_cast<cudf::size_type>(colchunks.size());

    // Binary search the the column chunk index for this page
    auto const chunk_idx = static_cast<cudf::size_type>(
      cuda::std::distance(
        chunk_page_offsets,
        thrust::upper_bound(
          thrust::seq, chunk_page_offsets, chunk_page_offsets + num_chunks + 1, page_idx)) -
      1);

    // Check if the chunk index is valid
    if (chunk_idx < 0 or chunk_idx >= num_chunks) {
      set_error(static_cast<kernel_error::value_type>(decode_error::DATA_STREAM_OVERRUN),
                error_code);
      return;
    }

    byte_stream_s bs{};
    bs.ck   = colchunks[chunk_idx];
    bs.base = bs.cur = page_locations[page_idx];
    bs.end           = bs.ck.compressed_data + bs.ck.compressed_size;
    // Check if byte stream pointers are valid.
    if (bs.end < bs.cur) {
      set_error(static_cast<kernel_error::value_type>(decode_error::DATA_STREAM_OVERRUN),
                error_code);
      return;
    }
    bs.page.chunk_idx      = chunk_idx;
    bs.page.src_col_schema = bs.ck.src_col_schema;

    // Zero out the rest of the page header info
    zero_out_page_header_info(&bs);

    // bs.page.chunk_row not computed here and will be filled in later by
    // `fill_in_page_info()`.

    if (not parse_page_header_fn{}(&bs) or bs.page.compressed_page_size < 0) {
      set_error(static_cast<kernel_error::value_type>(decode_error::UNSUPPORTED_ENCODING),
                error_code);
      return;
    }
    if (not is_supported_encoding(bs.page.encoding)) {
      set_error(static_cast<kernel_error::value_type>(decode_error::UNSUPPORTED_ENCODING),
                error_code);
      return;
    }
    switch (bs.page_type) {
      case PageType::DATA_PAGE:
        // this computation is only valid for flat schemas. for nested schemas,
        // they will be recomputed in the preprocess step by examining repetition and
        // definition levels
        bs.page.num_rows = bs.page.num_input_values;
        break;
      case PageType::DATA_PAGE_V2:
        bs.page.flags |= PAGEINFO_FLAGS_V2;
        // V2 only uses RLE, so it was removed from the header
        bs.page.definition_level_encoding = Encoding::RLE;
        bs.page.repetition_level_encoding = Encoding::RLE;
        break;
      case PageType::DICTIONARY_PAGE: bs.page.flags |= PAGEINFO_FLAGS_DICTIONARY; break;
      default:
        set_error(static_cast<kernel_error::value_type>(decode_error::INVALID_PAGE_TYPE),
                  error_code);
        return;
    }

    bs.page.page_data   = const_cast<uint8_t*>(bs.cur);
    bs.page.kernel_mask = kernel_mask_for_page(bs.page, bs.ck);

    // Copy over the page info from byte stream
    pages[page_idx] = bs.page;
  }
};

/**
 * @brief Kernel for building dictionary index for the specified column chunks
 *
 * This function builds an index to point to each dictionary entry
 * (string format is 4-byte little-endian string length followed by character
 * data). The index is a 32-bit integer which contains the offset of each string
 * relative to the beginning of the dictionary page data.
 *
 * @param[in] chunks List of column chunks
 * @param[in] num_chunks Number of column chunks
 */
CUDF_KERNEL void __launch_bounds__(build_string_dict_index_block_size)
  build_string_dictionary_index_kernel(ColumnChunkDesc* chunks, int32_t num_chunks)
{
  auto constexpr num_warps_per_block = build_string_dict_index_block_size / cudf::detail::warp_size;
  __shared__ ColumnChunkDesc chunk_g[num_warps_per_block];

  auto const block  = cg::this_thread_block();
  auto const warp   = cg::tiled_partition<cudf::detail::warp_size>(block);
  int const lane_id = warp.thread_rank();
  int const chunk   = (cg::this_grid().block_rank() * num_warps_per_block) + warp.meta_group_rank();
  ColumnChunkDesc* const ck = &chunk_g[warp.meta_group_rank()];
  if (chunk < num_chunks and lane_id == 0) *ck = chunks[chunk];
  block.sync();

  if (chunk >= num_chunks) { return; }
  if (!lane_id && ck->num_dict_pages > 0 && ck->str_dict_index) {
    // Data type to describe a string
    string_index_pair* dict_index = ck->str_dict_index;
    uint8_t const* dict           = ck->dict_page->page_data;
    int dict_size                 = ck->dict_page->uncompressed_page_size;
    int num_entries               = ck->dict_page->num_input_values;
    int pos = 0, cur = 0;
    for (int i = 0; i < num_entries; i++) {
      int len = 0;
      if (ck->physical_type == Type::FIXED_LEN_BYTE_ARRAY) {
        if (cur + ck->type_length <= dict_size) {
          len = ck->type_length;
          pos = cur;
          cur += len;
        } else {
          cur = dict_size;
        }
      } else {
        if (cur + 4 <= dict_size) {
          len =
            dict[cur + 0] | (dict[cur + 1] << 8) | (dict[cur + 2] << 16) | (dict[cur + 3] << 24);
          if (len >= 0 && cur + 4 + len <= dict_size) {
            pos = cur + 4;
            cur = pos + len;
          } else {
            cur = dict_size;
          }
        }
      }
      // TODO: Could store 8 entries in shared mem, then do a single warp-wide store
      dict_index[i].first  = reinterpret_cast<char const*>(dict + pos);
      dict_index[i].second = len;
    }
  }
}

}  // namespace

void count_page_headers(cudf::detail::hostdevice_span<ColumnChunkDesc> chunks,
                        kernel_error::pointer error_code,
                        rmm::cuda_stream_view stream)
{
  static_assert(count_page_headers_block_size % cudf::detail::warp_size == 0,
                "Block size for decode page headers kernel must be a multiple of warp size");

  // Use CPU implementation for small number of chunks
  if (chunks.size() <= cpu_decode_threshold) {
    // Ensure host has the latest data
    chunks.device_to_host_async(stream);
    stream.synchronize();

    // Copy compressed data for each chunk to host memory
    std::vector<std::vector<uint8_t>> h_compressed_data(chunks.size());
    std::vector<uint8_t const*> original_ptrs(chunks.size());

    for (size_t i = 0; i < chunks.size(); ++i) {
      original_ptrs[i] = chunks[i].compressed_data;
      h_compressed_data[i].resize(chunks[i].compressed_size);
      CUDF_CUDA_TRY(cudaMemcpyAsync(h_compressed_data[i].data(),
                                    chunks[i].compressed_data,
                                    chunks[i].compressed_size,
                                    cudaMemcpyDeviceToHost,
                                    stream.value()));
      chunks[i].compressed_data = h_compressed_data[i].data();
    }
    stream.synchronize();

    kernel_error::value_type error = 0;
    for (size_t i = 0; i < chunks.size(); ++i) {
      count_page_headers_chunk_cpu(chunks[i], &error);
    }
    if (error != 0) { set_error(error, error_code); }

    // Restore original device pointers before copying back
    for (size_t i = 0; i < chunks.size(); ++i) {
      chunks[i].compressed_data = original_ptrs[i];
    }

    // Copy updated chunks back to device
    chunks.host_to_device_async(stream);
    return;
  }

  auto constexpr num_warps_per_block = count_page_headers_block_size / cudf::detail::warp_size;
  auto const num_blocks              = cudf::util::div_rounding_up_unsafe<cudf::size_type>(
    chunks.size(), num_warps_per_block);  // 1 warp per chunk

  dim3 dim_block(count_page_headers_block_size, 1);
  dim3 dim_grid(num_blocks, 1);

  count_page_headers_kernel<<<dim_grid, dim_block, 0, stream.value()>>>(chunks, error_code);
}

void decode_page_headers(cudf::detail::hostdevice_span<ColumnChunkDesc const> chunks,
                         chunk_page_info* chunk_pages,
                         kernel_error::pointer error_code,
                         rmm::cuda_stream_view stream)
{
  static_assert(decode_page_headers_block_size % cudf::detail::warp_size == 0,
                "Block size for decode page headers kernel must be a multiple of warp size");

  auto const num_chunks = static_cast<cudf::size_type>(chunks.size());

  // Use CPU implementation for small number of chunks
  if (num_chunks <= cpu_decode_threshold) {
    // Copy chunks and chunk_pages to host (read-only for chunks)
    std::vector<ColumnChunkDesc> h_chunks(num_chunks);
    CUDF_CUDA_TRY(cudaMemcpyAsync(h_chunks.data(),
                                  chunks.device_ptr(),
                                  num_chunks * sizeof(ColumnChunkDesc),
                                  cudaMemcpyDeviceToHost,
                                  stream.value()));

    std::vector<chunk_page_info> h_chunk_pages(num_chunks);
    CUDF_CUDA_TRY(cudaMemcpyAsync(h_chunk_pages.data(),
                                  chunk_pages,
                                  num_chunks * sizeof(chunk_page_info),
                                  cudaMemcpyDeviceToHost,
                                  stream.value()));
    stream.synchronize();

    // Copy compressed data for each chunk to host memory
    std::vector<std::vector<uint8_t>> h_compressed_data(num_chunks);

    for (cudf::size_type i = 0; i < num_chunks; ++i) {
      h_compressed_data[i].resize(h_chunks[i].compressed_size);
      CUDF_CUDA_TRY(cudaMemcpyAsync(h_compressed_data[i].data(),
                                    h_chunks[i].compressed_data,
                                    h_chunks[i].compressed_size,
                                    cudaMemcpyDeviceToHost,
                                    stream.value()));
      h_chunks[i].compressed_data = h_compressed_data[i].data();
    }
    stream.synchronize();

    // Copy PageInfo arrays for each chunk to host memory
    std::vector<std::vector<PageInfo>> h_page_info(num_chunks);
    std::vector<PageInfo*> original_page_ptrs(num_chunks);
    std::vector<uint8_t const*> original_compressed_data_ptrs(num_chunks);

    for (cudf::size_type i = 0; i < num_chunks; ++i) {
      auto const num_pages  = h_chunks[i].num_data_pages + h_chunks[i].num_dict_pages;
      original_page_ptrs[i] = h_chunk_pages[i].pages;
      // Save the original device pointer for compressed_data before we replaced it
      CUDF_CUDA_TRY(cudaMemcpyAsync(&original_compressed_data_ptrs[i],
                                    &chunks.device_ptr()[i].compressed_data,
                                    sizeof(uint8_t const*),
                                    cudaMemcpyDeviceToHost,
                                    stream.value()));
      h_page_info[i].resize(num_pages);
      // Note: The pages array might not be initialized yet, so we just allocate space
      // If it's already allocated on device, we could copy it, but for decode we're writing to it
      h_chunk_pages[i].pages = h_page_info[i].data();
    }
    stream.synchronize();

    kernel_error::value_type error = 0;
    for (cudf::size_type i = 0; i < num_chunks; ++i) {
      decode_page_headers_chunk_cpu(
        h_chunks[i], h_chunk_pages[i].pages, i, &error, original_compressed_data_ptrs[i]);
    }

    // Restore original device pointers and copy PageInfo arrays back to device
    for (cudf::size_type i = 0; i < num_chunks; ++i) {
      auto const num_pages = h_chunks[i].num_data_pages + h_chunks[i].num_dict_pages;
      CUDF_CUDA_TRY(cudaMemcpyAsync(original_page_ptrs[i],
                                    h_page_info[i].data(),
                                    num_pages * sizeof(PageInfo),
                                    cudaMemcpyHostToDevice,
                                    stream.value()));
      h_chunk_pages[i].pages = original_page_ptrs[i];
    }

    // Copy results back to device (only chunk_pages, chunks are read-only)
    CUDF_CUDA_TRY(cudaMemcpyAsync(chunk_pages,
                                  h_chunk_pages.data(),
                                  num_chunks * sizeof(chunk_page_info),
                                  cudaMemcpyHostToDevice,
                                  stream.value()));

    if (error != 0) { set_error(error, error_code); }
    return;
  }

  auto constexpr num_warps_per_block = decode_page_headers_block_size / cudf::detail::warp_size;
  auto const num_blocks =
    cudf::util::div_rounding_up_unsafe(num_chunks, num_warps_per_block);  // 1 warp per chunk

  dim3 dim_block(decode_page_headers_block_size, 1);
  dim3 dim_grid(num_blocks, 1);

  decode_page_headers_kernel<<<dim_grid, dim_block, 0, stream.value()>>>(
    chunks, chunk_pages, error_code);
}

void decode_page_headers_with_pgidx(cudf::device_span<ColumnChunkDesc const> chunks,
                                    cudf::device_span<PageInfo> pages,
                                    uint8_t** page_locations,
                                    size_type* chunk_page_offsets,
                                    kernel_error::pointer error_code,
                                    rmm::cuda_stream_view stream)
{
  thrust::for_each(rmm::exec_policy_nosync(stream),
                   thrust::counting_iterator(0),
                   thrust::counting_iterator<cudf::size_type>(pages.size()),
                   decode_page_headers_with_pgidx_fn{.colchunks          = chunks,
                                                     .pages              = pages,
                                                     .page_locations     = page_locations,
                                                     .chunk_page_offsets = chunk_page_offsets,
                                                     .error_code         = error_code});
}

void build_string_dictionary_index(ColumnChunkDesc* chunks,
                                   int32_t num_chunks,
                                   rmm::cuda_stream_view stream)
{
  static_assert(
    build_string_dict_index_block_size % cudf::detail::warp_size == 0,
    "Block size for build string dictionary index kernel must be a multiple of warp size");
  auto constexpr num_warps_per_block = build_string_dict_index_block_size / cudf::detail::warp_size;
  auto const num_blocks =
    cudf::util::div_rounding_up_unsafe(num_chunks, num_warps_per_block);  // 1 warp per chunk

  dim3 dim_block(build_string_dict_index_block_size, 1);
  dim3 dim_grid(num_blocks, 1);

  build_string_dictionary_index_kernel<<<dim_grid, dim_block, 0, stream.value()>>>(chunks,
                                                                                   num_chunks);
}

}  // namespace cudf::io::parquet::detail
