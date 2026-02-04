/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "error.hpp"
#include "parquet_gpu.hpp"

namespace cudf::io::parquet::detail {

/**
 * @brief Minimal thrift implementation for parsing page headers
 *
 * See: https://github.com/apache/thrift/blob/master/doc/specs/thrift-compact-protocol.md
 */
struct byte_stream_s {
  uint8_t const* cur{};
  uint8_t const* end{};
  uint8_t const* base{};
  // Parsed symbols
  PageType page_type{};
  PageInfo page{};
  ColumnChunkDesc ck{};
};

/**
 * @brief Get current byte from the byte stream
 *
 * @param bs Byte stream
 *
 * @return Current byte pointed to by the byte stream
 */
CUDF_HOST_DEVICE inline unsigned int getb(byte_stream_s* bs)
{
  return (bs->cur < bs->end) ? *bs->cur++ : 0;
}

CUDF_HOST_DEVICE inline void skip_bytes(byte_stream_s* bs, size_t bytecnt)
{
  bytecnt = min(bytecnt, (size_t)(bs->end - bs->cur));
  bs->cur += bytecnt;
}

/**
 * @brief Decode unsigned integer from a byte stream using VarInt encoding
 *
 * Concatenate least significant 7 bits of each byte to form a 32 bit
 * integer. Most significant bit of each byte indicates if more bytes
 * are to be used to form the number.
 *
 * @param bs Byte stream
 *
 * @return Decoded 32 bit integer
 */
CUDF_HOST_DEVICE inline uint32_t get_u32(byte_stream_s* bs)
{
  uint32_t v = 0, l = 0, c;
  do {
    c = getb(bs);
    v |= (c & 0x7f) << l;
    l += 7;
  } while (c & 0x80);
  return v;
}

/**
 * @brief Decode signed integer from a byte stream using zigzag encoding
 *
 * The number n encountered in a byte stream translates to
 * -1^(n%2) * ceil(n/2), with the exception of 0 which remains the same.
 * i.e. 0, 1, 2, 3, 4, 5 etc convert to 0, -1, 1, -2, 2 respectively.
 *
 * @param bs Byte stream
 *
 * @return Decoded 32 bit integer
 */
CUDF_HOST_DEVICE inline int32_t get_i32(byte_stream_s* bs)
{
  uint32_t u = get_u32(bs);
  return (int32_t)((u >> 1u) ^ -(int32_t)(u & 1));
}

/**
 * @brief Skip a struct field in the byte stream
 *
 * @param bs Byte stream
 * @param field_type Field type
 */
CUDF_HOST_DEVICE inline void skip_struct_field(byte_stream_s* bs, int field_type)
{
  int struct_depth = 0;
  int rep_cnt      = 0;

  do {
    if (rep_cnt != 0) {
      rep_cnt--;
    } else if (struct_depth != 0) {
      unsigned int c;
      do {
        c = getb(bs);
        if (!c) --struct_depth;
      } while (!c && struct_depth);
      if (!struct_depth) break;
      field_type = c & 0xf;
      if (!(c & 0xf0)) get_i32(bs);
    }
    switch (static_cast<FieldType>(field_type)) {
      case FieldType::BOOLEAN_TRUE:
      case FieldType::BOOLEAN_FALSE: break;
      case FieldType::I16:
      case FieldType::I32:
      case FieldType::I64: get_u32(bs); break;
      case FieldType::I8: skip_bytes(bs, 1); break;
      case FieldType::DOUBLE: skip_bytes(bs, 8); break;
      case FieldType::BINARY: skip_bytes(bs, get_u32(bs)); break;
      case FieldType::LIST:
      case FieldType::SET: {  // NOTE: skipping a list of lists is not handled
        auto const c = getb(bs);
        int n        = c >> 4;
        if (n == 0xf) { n = get_u32(bs); }
        field_type = c & 0xf;
        if (static_cast<FieldType>(field_type) == FieldType::STRUCT) {
          struct_depth += n;
        } else {
          rep_cnt = n;
        }
      } break;
      case FieldType::STRUCT: struct_depth++; break;
      default: break;  // UUID, MAP are not supported yet
    }
  } while (rep_cnt || struct_depth);
}

/**
 * @brief Check if the column chunk has nesting
 *
 * @param chunk Column chunk
 *
 * @return True if the column chunk has nesting
 */
CUDF_HOST_DEVICE inline bool is_nested(ColumnChunkDesc const& chunk)
{
  return chunk.max_nesting_depth > 1;
}

/**
 * @brief Check if the column chunk is a list type
 *
 * @param chunk Column chunk
 *
 * @return True if the column chunk is a list type
 */
CUDF_HOST_DEVICE inline bool is_list(ColumnChunkDesc const& chunk)
{
  return chunk.max_level[level_type::REPETITION] > 0;
}

/**
 * @brief Check if the column chunk is a byte array type
 *
 * @param chunk Column chunk
 *
 * @return True if the column chunk is a byte array type
 */
CUDF_HOST_DEVICE inline bool is_byte_array(ColumnChunkDesc const& chunk)
{
  return chunk.physical_type == Type::BYTE_ARRAY;
}

/**
 * @brief Check if the column chunk is a boolean type
 *
 * @param chunk Column chunk
 *
 * @return True if the column chunk is a boolean type
 */
CUDF_HOST_DEVICE inline bool is_boolean(ColumnChunkDesc const& chunk)
{
  return chunk.physical_type == Type::BOOLEAN;
}

/**
 * @brief Determine which decode kernel to run for the given page.
 *
 * @param page The page to decode
 * @param chunk Column chunk the page belongs to
 * @return `kernel_mask_bits` value for the given page
 */
CUDF_HOST_DEVICE inline decode_kernel_mask kernel_mask_for_page(PageInfo const& page,
                                                                ColumnChunkDesc const& chunk)
{
  if (page.flags & PAGEINFO_FLAGS_DICTIONARY) { return decode_kernel_mask::NONE; }

  if (page.encoding == Encoding::DELTA_BINARY_PACKED) {
    return decode_kernel_mask::DELTA_BINARY;
  } else if (page.encoding == Encoding::DELTA_BYTE_ARRAY) {
    return decode_kernel_mask::DELTA_BYTE_ARRAY;
  } else if (page.encoding == Encoding::DELTA_LENGTH_BYTE_ARRAY) {
    return decode_kernel_mask::DELTA_LENGTH_BA;
  } else if (is_boolean(chunk)) {
    return is_list(chunk)     ? decode_kernel_mask::BOOLEAN_LIST
           : is_nested(chunk) ? decode_kernel_mask::BOOLEAN_NESTED
                              : decode_kernel_mask::BOOLEAN;
  }

  if (is_string_col(chunk)) {
    // check for string before byte_stream_split so FLBA will go to the right kernel
    if (page.encoding == Encoding::PLAIN) {
      return is_list(chunk)     ? decode_kernel_mask::STRING_LIST
             : is_nested(chunk) ? decode_kernel_mask::STRING_NESTED
                                : decode_kernel_mask::STRING;
    } else if (page.encoding == Encoding::PLAIN_DICTIONARY ||
               page.encoding == Encoding::RLE_DICTIONARY) {
      return is_list(chunk)     ? decode_kernel_mask::STRING_DICT_LIST
             : is_nested(chunk) ? decode_kernel_mask::STRING_DICT_NESTED
                                : decode_kernel_mask::STRING_DICT;
    } else if (page.encoding == Encoding::BYTE_STREAM_SPLIT) {
      return is_list(chunk)     ? decode_kernel_mask::STRING_STREAM_SPLIT_LIST
             : is_nested(chunk) ? decode_kernel_mask::STRING_STREAM_SPLIT_NESTED
                                : decode_kernel_mask::STRING_STREAM_SPLIT;
    }
  }

  if (!is_byte_array(chunk)) {
    if (page.encoding == Encoding::PLAIN) {
      return is_list(chunk)     ? decode_kernel_mask::FIXED_WIDTH_NO_DICT_LIST
             : is_nested(chunk) ? decode_kernel_mask::FIXED_WIDTH_NO_DICT_NESTED
                                : decode_kernel_mask::FIXED_WIDTH_NO_DICT;
    } else if (page.encoding == Encoding::PLAIN_DICTIONARY ||
               page.encoding == Encoding::RLE_DICTIONARY) {
      return is_list(chunk)     ? decode_kernel_mask::FIXED_WIDTH_DICT_LIST
             : is_nested(chunk) ? decode_kernel_mask::FIXED_WIDTH_DICT_NESTED
                                : decode_kernel_mask::FIXED_WIDTH_DICT;
    } else if (page.encoding == Encoding::BYTE_STREAM_SPLIT) {
      return is_list(chunk)     ? decode_kernel_mask::BYTE_STREAM_SPLIT_FIXED_WIDTH_LIST
             : is_nested(chunk) ? decode_kernel_mask::BYTE_STREAM_SPLIT_FIXED_WIDTH_NESTED
                                : decode_kernel_mask::BYTE_STREAM_SPLIT_FIXED_WIDTH_FLAT;
    }
  }

  if (page.encoding == Encoding::BYTE_STREAM_SPLIT) {
    return decode_kernel_mask::BYTE_STREAM_SPLIT;
  }

  // non-string, non-delta, non-split_stream
  return decode_kernel_mask::GENERAL;
}

/**
 * @brief Functor to set value to bool read from byte stream
 *
 * @return True if field type is not bool
 */
struct ParquetFieldBool {
  int field;
  bool& val;

  CUDF_HOST_DEVICE ParquetFieldBool(int f, bool& v) : field(f), val(v) {}

  CUDF_HOST_DEVICE inline bool operator()(byte_stream_s* bs, int field_type)
  {
    val = static_cast<FieldType>(field_type) == FieldType::BOOLEAN_TRUE;
    return not(static_cast<FieldType>(field_type) == FieldType::BOOLEAN_TRUE or
               static_cast<FieldType>(field_type) == FieldType::BOOLEAN_FALSE);
  }
};

/**
 * @brief Functor to set value to 32 bit integer read from byte stream
 *
 * @return True if field type is not int32
 */
struct ParquetFieldInt32 {
  int field;
  int32_t& val;

  CUDF_HOST_DEVICE ParquetFieldInt32(int f, int32_t& v) : field(f), val(v) {}

  CUDF_HOST_DEVICE inline bool operator()(byte_stream_s* bs, int field_type)
  {
    val = get_i32(bs);
    return (static_cast<FieldType>(field_type) != FieldType::I32);
  }
};

/**
 * @brief Functor to set value to enum read from byte stream
 *
 * @return True if field type is not int32
 */
template <typename Enum>
struct ParquetFieldEnum {
  int field;
  Enum& val;

  CUDF_HOST_DEVICE ParquetFieldEnum(int f, Enum& v) : field(f), val(v) {}

  CUDF_HOST_DEVICE inline bool operator()(byte_stream_s* bs, int field_type)
  {
    val = static_cast<Enum>(get_i32(bs));
    return (static_cast<FieldType>(field_type) != FieldType::I32);
  }
};

/**
 * @brief Functor to run operator on byte stream
 *
 * @return True if field type is not struct type or if the calling operator
 * fails
 */
template <typename Operator>
struct ParquetFieldStruct {
  int field;
  Operator op;

  CUDF_HOST_DEVICE ParquetFieldStruct(int f) : field(f) {}

  CUDF_HOST_DEVICE inline bool operator()(byte_stream_s* bs, int field_type)
  {
    return ((static_cast<FieldType>(field_type) != FieldType::STRUCT) || !op(bs));
  }
};

/**
 * @brief Functor to run an operator
 *
 * The purpose of this functor is to replace a switch case. If the field in
 * the argument is equal to the field specified in any element of the tuple
 * of operators then it is run with the byte stream and field type arguments.
 *
 * If the field does not match any of the functors then skip_struct_field is
 * called over the byte stream.
 *
 * @return Return value of the selected operator or false if no operator
 * matched the field value
 */
template <int index>
struct FunctionSwitchImpl {
  template <typename... Operator>
  CUDF_HOST_DEVICE static inline bool run(byte_stream_s* bs,
                                          int field_type,
                                          int const& field,
                                          cuda::std::tuple<Operator...>& ops)
  {
    if (field == cuda::std::get<index>(ops).field) {
      return cuda::std::get<index>(ops)(bs, field_type);
    } else {
      return FunctionSwitchImpl<index - 1>::run(bs, field_type, field, ops);
    }
  }
};

template <>
struct FunctionSwitchImpl<0> {
  template <typename... Operator>
  CUDF_HOST_DEVICE static inline bool run(byte_stream_s* bs,
                                          int field_type,
                                          int const& field,
                                          cuda::std::tuple<Operator...>& ops)
  {
    if (field == cuda::std::get<0>(ops).field) {
      return cuda::std::get<0>(ops)(bs, field_type);
    } else {
      skip_struct_field(bs, field_type);
      return false;
    }
  }
};

/**
 * @brief Function to parse page header based on the tuple of functors provided
 *
 * Bytes are read from the byte stream and the field delta and field type are
 * matched up against user supplied reading functors. If they match then the
 * corresponding values are written to references pointed to by the functors.
 *
 * @return Returns false if an unexpected field is encountered while reading
 * byte stream. Otherwise true is returned.
 */
template <typename... Operator>
CUDF_HOST_DEVICE inline bool parse_header(cuda::std::tuple<Operator...>& op, byte_stream_s* bs)
{
  constexpr int index = cuda::std::tuple_size<cuda::std::tuple<Operator...>>::value - 1;
  int field           = 0;
  while (true) {
    auto const current_byte = getb(bs);
    if (!current_byte) break;
    int const field_delta = current_byte >> 4;
    int const field_type  = current_byte & 0xf;
    field                 = field_delta ? field + field_delta : get_i32(bs);
    bool exit_function    = FunctionSwitchImpl<index>::run(bs, field_type, field, op);
    if (exit_function) { return false; }
  }
  return true;
}

/**
 * @brief Functor to parse v1 data page header
 *
 * @param bs Byte stream
 *
 * @return True if the data page header is parsed successfully
 */
struct parse_data_page_header_fn {
  CUDF_HOST_DEVICE bool operator()(byte_stream_s* bs)
  {
    auto op =
      cuda::std::make_tuple(ParquetFieldInt32(1, bs->page.num_input_values),
                            ParquetFieldEnum<Encoding>(2, bs->page.encoding),
                            ParquetFieldEnum<Encoding>(3, bs->page.definition_level_encoding),
                            ParquetFieldEnum<Encoding>(4, bs->page.repetition_level_encoding));
    return parse_header(op, bs);
  }
};

/**
 * @brief Functor to parse dictionary page header
 *
 * @param bs Byte stream
 *
 * @return True if the dictionary page header is parsed successfully
 */
struct parse_dictionary_page_header_fn {
  CUDF_HOST_DEVICE bool operator()(byte_stream_s* bs)
  {
    auto op = cuda::std::make_tuple(ParquetFieldInt32(1, bs->page.num_input_values),
                                    ParquetFieldEnum<Encoding>(2, bs->page.encoding));
    return parse_header(op, bs);
  }
};

/**
 * @brief Functor to parse V2 data page header
 *
 * @param bs Byte stream
 *
 * @return True if the data page header V2 is parsed successfully
 */
struct parse_data_page_header_v2_fn {
  CUDF_HOST_DEVICE bool operator()(byte_stream_s* bs)
  {
    auto op =
      cuda::std::make_tuple(ParquetFieldInt32(1, bs->page.num_input_values),
                            ParquetFieldInt32(2, bs->page.num_nulls),
                            ParquetFieldInt32(3, bs->page.num_rows),
                            ParquetFieldEnum<Encoding>(4, bs->page.encoding),
                            ParquetFieldInt32(5, bs->page.lvl_bytes[level_type::DEFINITION]),
                            ParquetFieldInt32(6, bs->page.lvl_bytes[level_type::REPETITION]),
                            ParquetFieldBool(7, bs->page.is_compressed));
    return parse_header(op, bs);
  }
};

/**
 * @brief Functor to parse page header from byte stream
 *
 * @param bs Byte stream
 *
 * @return True if the page header is parsed successfully
 */
struct parse_page_header_fn {
  CUDF_HOST_DEVICE bool operator()(byte_stream_s* bs)
  {
    auto op = cuda::std::make_tuple(ParquetFieldEnum<PageType>(1, bs->page_type),
                                    ParquetFieldInt32(2, bs->page.uncompressed_page_size),
                                    ParquetFieldInt32(3, bs->page.compressed_page_size),
                                    ParquetFieldStruct<parse_data_page_header_fn>(5),
                                    ParquetFieldStruct<parse_dictionary_page_header_fn>(7),
                                    ParquetFieldStruct<parse_data_page_header_v2_fn>(8));
    return parse_header(op, bs);
  }
};

/**
 * @brief Zero out page header info
 *
 * @param bs Byte stream
 */
CUDF_HOST_DEVICE inline void zero_out_page_header_info(byte_stream_s* bs)
{
  // this computation is only valid for flat schemas. for nested schemas,
  // they will be recomputed in the preprocess step by examining repetition and
  // definition levels
  bs->page.chunk_row            = 0;
  bs->page.num_rows             = 0;
  bs->page.is_num_rows_adjusted = false;
  bs->page.skipped_values       = -1;
  bs->page.skipped_leaf_values  = 0;
  bs->page.str_bytes            = 0;
  bs->page.str_bytes_from_index = 0;
  bs->page.num_valids           = 0;
  bs->page.start_val            = 0;
  bs->page.end_val              = 0;
  bs->page.has_page_index       = false;
  bs->page.temp_string_size     = 0;
  bs->page.temp_string_buf      = nullptr;
  bs->page.kernel_mask          = decode_kernel_mask::NONE;
  bs->page.is_compressed        = true;
  bs->page.flags                = 0;
  bs->page.str_bytes_all        = 0;
  // zero out V2 info
  bs->page.num_nulls                         = 0;
  bs->page.lvl_bytes[level_type::DEFINITION] = 0;
  bs->page.lvl_bytes[level_type::REPETITION] = 0;
}

}  // namespace cudf::io::parquet::detail
