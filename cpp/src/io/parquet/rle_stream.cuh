/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "parquet_gpu.hpp"

#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>

namespace cudf::io::parquet::detail {

template <int num_threads>
constexpr int rle_stream_required_run_buffer_size()
{
  constexpr int num_rle_stream_decode_warps = (num_threads / cudf::detail::warp_size) - 1;
  return (num_rle_stream_decode_warps * 2);
}

/**
 * @brief Read a 32-bit varint integer
 *
 * @param[in,out] cur The current data position, updated after the read
 * @param[in] end The end data position
 *
 * @return The 32-bit value read
 */
inline __device__ uint32_t get_vlq32(uint8_t const*& cur, uint8_t const* end)
{
  uint32_t v = *cur++;
  if (v >= 0x80 && cur < end) {
    v = (v & 0x7f) | ((*cur++) << 7);
    if (v >= (0x80 << 7) && cur < end) {
      v = (v & ((0x7f << 7) | 0x7f)) | ((*cur++) << 14);
      if (v >= (0x80 << 14) && cur < end) {
        v = (v & ((0x7f << 14) | (0x7f << 7) | 0x7f)) | ((*cur++) << 21);
        if (v >= (0x80 << 21) && cur < end) {
          v = (v & ((0x7f << 21) | (0x7f << 14) | (0x7f << 7) | 0x7f)) | ((*cur++) << 28);
        }
      }
    }
  }
  return v;
}

/**
 * @brief RLE run decode function per warp.
 *
 * @param output output data buffer
 * @param literal_run true if literal run, false if repeated run
 * @param run_start_pos beginning of data for RLE run
 * @param stream_end pointer to the end of data for RLE run
 * @param run_output_index absolute output position for this run
 * @param run_offset_count offset after run_output_index this call to decode starts outputting at
 * @param count length that will be decoded in this decode call, truncated to fit output buffer
 * @param num_bits_per_value bits needed to encode max values in the run (definition, dictionary)
 * @param lane warp lane that is executing this decode call
 */
template <typename level_t, int max_output_values>
__device__ inline void decode(level_t* const output,
                              bool const literal_run,
                              uint8_t const* const run_start_pos,
                              uint8_t const* const stream_end,
                              int const run_output_index,
                              int const run_offset_count,
                              int const count,
                              int num_bits_per_value,
                              int lane)
{
  // local output_index for this `decode` call.
  int decode_output_index = 0;
  int remaining_count     = count;

  // for bitpacked/literal runs, total size is always a multiple of 8. so we need to take care if
  // we are not starting/ending exactly on a run boundary
  uint8_t const* current_run_pos;
  int data_value;
  if (literal_run) {
    int const effective_offset = cudf::util::round_down_safe(run_offset_count, 8);
    int const lead_values      = (run_offset_count - effective_offset);
    decode_output_index -= lead_values;
    remaining_count += lead_values;
    current_run_pos = run_start_pos + ((effective_offset >> 3) * num_bits_per_value);
  } else {
    // this is a repeated run, compute the repeated value
    data_value = run_start_pos[0];
    if constexpr (sizeof(level_t) > 1) {
      if (num_bits_per_value > 8) {
        data_value |= run_start_pos[1] << 8;
        if constexpr (sizeof(level_t) > 2) {
          if (num_bits_per_value > 16) {
            data_value |= run_start_pos[2] << 16;
            if (num_bits_per_value > 24) { data_value |= run_start_pos[3] << 24; }
          }
        }
      }
    }
  }

  // process
  while (remaining_count > 0) {
    int const batch_count = min(32, remaining_count);

    // if this is a literal run. each thread computes its own data_value
    if (literal_run) {
      int const batch_num_bytes = (batch_count + 7) >> 3;
      if (lane < batch_count) {
        int bitpos                = lane * num_bits_per_value;
        uint8_t const* thread_run_pos = current_run_pos + (bitpos >> 3);
        bitpos &= 7;
        data_value = 0;
        if (thread_run_pos < stream_end) { data_value = thread_run_pos[0]; }
        thread_run_pos++;
        if (num_bits_per_value > 8 - bitpos && thread_run_pos < stream_end) {
          data_value |= thread_run_pos[0] << 8;
          thread_run_pos++;
          if (num_bits_per_value > 16 - bitpos && thread_run_pos < stream_end) {
            data_value |= thread_run_pos[0] << 16;
            thread_run_pos++;
            if (num_bits_per_value > 24 - bitpos && thread_run_pos < stream_end) { data_value |= thread_run_pos[0] << 24; }
          }
        }
        data_value = (data_value >> bitpos) & ((1 << num_bits_per_value) - 1);
      }

      current_run_pos += batch_num_bytes * num_bits_per_value;
    }

    // store data_value
    if (lane < batch_count && (lane + decode_output_index) >= 0) {
      auto const idx = lane + run_output_index + run_offset_count + decode_output_index;
      output[rolling_index<max_output_values>(idx)] = data_value;
    }
    remaining_count -= batch_count;
    decode_output_index += batch_count;
  }
}

// a single rle run. may be broken up into multiple rle_batches
struct rle_run {
  int count;        // total count of the run
  int output_index;  // absolute position of this run w.r.t output
  uint8_t const* start_pos;
  bool literal_run;  // true if literal run, false if repeated run
  int remaining_count;  // number of output items remaining to be decoded
};

// a stream of rle_runs
template <typename level_t, int decode_threads, int max_output_values>
struct rle_stream {
  static constexpr int num_rle_stream_decode_threads = decode_threads;
  // the -1 here is for the look-ahead warp that fills in the list of runs to be decoded
  // in an overlapped manner. so if we had 16 total warps:
  // - warp 0 would be filling in batches of runs to be processed
  // - warps 1-15 would be decoding the previous batch of runs generated
  static constexpr int num_rle_stream_decode_warps =
    (num_rle_stream_decode_threads / cudf::detail::warp_size) - 1;

  static constexpr int run_buffer_size = rle_stream_required_run_buffer_size<decode_threads>();

  int num_bits_per_value;
  uint8_t const* stream_next_run_start;
  uint8_t const* stream_end;

  int total_values;
  int num_values_processed;

  level_t* output;

  rle_run* runs;

  int output_index;

  int run_fill_index;
  int decode_run_index;

  __device__ rle_stream(rle_run* _runs) : runs(_runs) {}

  __device__ inline bool is_last_decode_warp(int warp_id)
  {
    return warp_id == num_rle_stream_decode_warps;
  }

  __device__ void init(int _num_bits_per_value,
                       uint8_t const* _stream_start,
                       uint8_t const* _stream_end,
                       level_t* _output,
                       int _total_values)
  {
    num_bits_per_value = _num_bits_per_value;
    stream_next_run_start        = _stream_start;
    stream_end        = _stream_end;

    output = _output;

    output_index = 0;

    total_values = _total_values;
    num_values_processed   = 0;
    run_fill_index   = 0;
    decode_run_index = -1;  // signals the first iteration. Nothing to decode.
  }

  __device__ inline int get_rle_run_info(rle_run& run)
  {
    uint8_t const* start_pos     = stream_next_run_start;
    int level_run = get_vlq32(start_pos, stream_end);
    bool literal_run = is_literal_run(level_run);

    // run_bytes includes the header size
    int run_bytes = start_pos - stream_next_run_start;
    int run_count;
    if (literal_run) {
      // from the parquet spec: literal runs always come in multiples of 8 values.
      run_count = (level_run >> 1) * 8;
      run_bytes += util::div_rounding_up_unsafe(run_count * num_bits_per_value, 8);
    } else {
      // repeated value run
      run_count = (level_run >> 1);
      run_bytes += util::div_rounding_up_unsafe(num_bits_per_value, 8);
    }
    run.count = run_count;
    run.literal_run = literal_run;
    run.start_pos = start_pos;

    return run_bytes;
  }

  __device__ inline void fill_run_batch()
  {
    // decode_run_index == -1 means we are on the very first decode iteration for this stream.
    // In this first iteration we are filling up to half of the runs array to decode in the next
    // iteration. On subsequent iterations, decode_run_index >= 0 and we are going to fill as many run
    // slots available as we can, to fill up to the slot before decode_run_index. We are also always
    // bound by stream_end, making sure we stop decoding once we've reached the end of the stream.
    while (((decode_run_index == -1 && run_fill_index < num_rle_stream_decode_warps) ||
            run_fill_index < decode_run_index + run_buffer_size) &&
           stream_next_run_start < stream_end) {
      // Encoding::RLE
      // Pass by reference to fill the runs shared memory with the run data
      auto& run           = runs[rolling_index<run_buffer_size>(run_fill_index)];
      int const run_bytes = get_rle_run_info(run);

      run.remaining_count  = run.count;
      run.output_index = output_index;

      stream_next_run_start += run_bytes;
      output_index += run.count;
      run_fill_index++;
    }
  }

  __device__ inline int decode_next(int t, int count)
  {
    int const output_count = min(count, total_values - num_values_processed);
    // special case. if num_bits_per_value == 0, just return all zeros. this should tremendously speed up
    // a very common case: columns with no nulls, especially if they are non-nested
    if (num_bits_per_value == 0) {
      int written = 0;
      while (written < output_count) {
        int const batch_size = min(num_rle_stream_decode_threads, output_count - written);
        if (t < batch_size) { output[rolling_index<max_output_values>(written + t)] = 0; }
        written += batch_size;
      }
      num_values_processed += output_count;
      return output_count;
    }

    // otherwise, full decode.
    int const warp_id        = t / cudf::detail::warp_size;
    int const warp_decode_id = warp_id - 1;
    int const warp_lane      = t % cudf::detail::warp_size;

    __shared__ int values_processed_shared;
    __shared__ int decode_run_index_shared;
    __shared__ int run_fill_index_shared;
    if (t == 0) {
      values_processed_shared = 0;
      decode_run_index_shared     = decode_run_index;
      run_fill_index_shared       = run_fill_index;
    }

    __syncthreads();

    run_fill_index = run_fill_index_shared;

    do {
      // protect against threads advancing past the end of this loop
      // and updating shared variables.
      __syncthreads();

      // warp 0 reads ahead and fills `runs` array to be decoded by remaining warps.
      if (warp_id == 0) {
        // fill the next set of runs. fill_runs will generally be the bottleneck for any
        // kernel that uses an rle_stream.
        if (warp_lane == 0) {
          fill_run_batch();
          if (decode_run_index == -1) {
            // first time, set it to the beginning of the buffer (rolled)
            decode_run_index        = 0;
            decode_run_index_shared = decode_run_index;
          }
          run_fill_index_shared = run_fill_index;
        }
      }
      // remaining warps decode the runs, starting on the second iteration of this. the pipeline of
      // runs is also persistent across calls to decode_next, so on the second call to decode_next,
      // this branch will start doing work immediately.
      // do/while loop (decode_run_index == -1 means "first iteration", so we should skip decoding)
      else if (decode_run_index >= 0 && decode_run_index + warp_decode_id < run_fill_index) {
        int const run_index = decode_run_index + warp_decode_id;
        auto& run           = runs[rolling_index<run_buffer_size>(run_index)];
        // this is the total amount (absolute) we will write in this invocation
        // of `decode_next`.
        int const max_count = num_values_processed + output_count;
        // run.output_index is absolute index, we start decoding
        // if it's supposed to fit in this call to `decode_next`.
        if (max_count > run.output_index) {
          int remaining_count        = run.remaining_count;
          int const run_offset_count = run.count - remaining_count;
          // last_run_index is the absolute position of the run, including
          // what was decoded last time.
          int const last_run_index = run.output_index + run_offset_count;

          // the amount we should process is the smallest of current remaining_count, or
          // space available in the output buffer (for that last run at the end of
          // a call to decode_next).
          int const batch_count = min(remaining_count, max_count - last_run_index);
          decode<level_t, max_output_values>(output,
                                             run.literal_run,
                                             run.start_pos,
                                             stream_end,
                                             run.output_index,
                                             run_offset_count,
                                             batch_count,
                                             num_bits_per_value,
                                             warp_lane);

          __syncwarp();
          if (warp_lane == 0) {
            // after writing this batch, are we at the end of the output buffer?
            auto const at_output_end = ((last_run_index + batch_count) == max_count);

            // update remaining_count for my warp
            remaining_count -= batch_count;
            // this is the last batch we will process this iteration if:
            // - either this run still has remaining values
            // - or it is consumed fully and its last index corresponds to output_count
            if (remaining_count > 0 || at_output_end) { values_processed_shared = output_count; }
            if (remaining_count == 0 && (at_output_end || is_last_decode_warp(warp_id))) {
              decode_run_index_shared = run_index + 1;
            }
            run.remaining_count = remaining_count;
          }
        }
      }
      __syncthreads();
      decode_run_index = decode_run_index_shared;
      run_fill_index = run_fill_index_shared;
    } while (values_processed_shared < output_count);

    num_values_processed += values_processed_shared;

    // valid for every thread
    return values_processed_shared;
  }

  __device__ inline int skip_runs(int target_count)
  {
    // we want to process all runs UP TO BUT NOT INCLUDING the run that overlaps with the skip
    // amount so threads spin like crazy on fill_run_batch(), skipping writing unnecessary run info.
    // then when it hits the one that matters, we don't process it at all and bail as if we never
    // started. basically we're setting up the rle_stream vars necessary to start fill_run_batch for
    // the first time
    while (stream_next_run_start < stream_end) {
      rle_run run;
      int run_bytes = get_rle_run_info(run);

      if ((output_index + run.count) > target_count) {
        return output_index;  // bail! we've reached the starting run
      }

      // skip this run
      output_index += run.count;
      stream_next_run_start += run_bytes;
    }

    return output_index;  // we skipped everything
  }

  __device__ inline int skip_decode(int t, int count)
  {
    int const output_count = min(count, total_values - num_values_processed);

    // if num_bits_per_value == 0, there's nothing to do
    // a very common case: columns with no nulls, especially if they are non-nested
    num_values_processed = (num_bits_per_value == 0) ? output_count : skip_runs(output_count);
    return num_values_processed;
  }

  __device__ inline int decode_next(int t) { return decode_next(t, max_output_values); }
};

}  // namespace cudf::io::parquet::detail
