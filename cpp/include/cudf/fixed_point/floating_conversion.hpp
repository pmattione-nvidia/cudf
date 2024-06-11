/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cudf/utilities/traits.hpp>

#include <cuda/std/limits>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>

#include <cstring>

namespace numeric {

/**
 * @addtogroup floating_conversion
 * @{
 * @file
 * @brief fixed_point <--> floating-point conversion functions.
 */

namespace detail {

/**
 * @brief Helper struct for getting and setting the components of a floating-point value
 *
 * @tparam FloatingType Type of floating-point value
 */
template <typename FloatingType, CUDF_ENABLE_IF(cuda::std::is_floating_point_v<FloatingType>)>
struct floating_converter {
  // This struct assumes we're working with IEEE 754 floating-point values.
  // Details on the IEEE-754 floating-point format:
  // Format: https://learn.microsoft.com/en-us/cpp/build/ieee-floating-point-representation
  // Float Visualizer: https://www.h-schmidt.net/FloatConverter/IEEE754.html
  static_assert(cuda::std::numeric_limits<FloatingType>::is_iec559, "Assumes IEEE 754");

  /// Unsigned int type with same size as floating type
  using IntegralType =
    cuda::std::conditional_t<cuda::std::is_same_v<FloatingType, float>, uint32_t, uint64_t>;

  // The high bit is the sign bit (0 for positive, 1 for negative).
  /// How many bits in the floating type
  static constexpr int num_floating_bits = sizeof(FloatingType) * CHAR_BIT;
  /// The index of the sign bit
  static constexpr int sign_bit_index = num_floating_bits - 1;
  /// The mask to select the sign bit
  static constexpr IntegralType sign_mask = (IntegralType(1) << sign_bit_index);

  // The low 23 / 52 bits (for float / double) are the mantissa.
  // The mantissa is normalized. There is an understood 1 bit to the left of the binary point.
  // The value of the mantissa is in the range [1, 2).
  /// # mantissa bits (-1 for understood bit)
  static constexpr int num_mantissa_bits = cuda::std::numeric_limits<FloatingType>::digits - 1;
  /// The mask for the understood bit
  static constexpr IntegralType understood_bit_mask = (IntegralType(1) << num_mantissa_bits);
  /// The mask to select the mantissa
  static constexpr IntegralType mantissa_mask = understood_bit_mask - 1;

  // And in between are the bits used to store the biased power-of-2 exponent.
  /// # exponents bits (-1 for sign bit)
  static constexpr int num_exponent_bits = num_floating_bits - num_mantissa_bits - 1;
  /// The mask for the exponents, unshifted
  static constexpr IntegralType unshifted_exponent_mask =
    (IntegralType(1) << num_exponent_bits) - 1;
  /// The mask to select the exponents
  static constexpr IntegralType exponent_mask = unshifted_exponent_mask << num_mantissa_bits;

  // To store positive and negative exponents as unsigned values, the stored value for
  // the power-of-2 is exponent + bias. The bias is 127 for floats and 1023 for doubles.
  /// 127 / 1023 for float / double
  static constexpr int exponent_bias = cuda::std::numeric_limits<FloatingType>::max_exponent - 1;

  /**
   * @brief Reinterpret the bits of a floating-point value as an integer
   *
   * @param floating The floating-point value to cast
   * @return An integer with bits identical to the input
   */
  CUDF_HOST_DEVICE inline static IntegralType bit_cast_to_integer(FloatingType floating)
  {
    // Convert floating to integer
    IntegralType integer_rep;
    memcpy(&integer_rep, &floating, sizeof(floating));
    return integer_rep;
  }

  /**
   * @brief Reinterpret the bits of an integer as floating-point value
   *
   * @param integer The integer to cast
   * @return A floating-point value with bits identical to the input
   */
  CUDF_HOST_DEVICE inline static FloatingType bit_cast_to_floating(IntegralType integer)
  {
    // Convert back to float
    FloatingType floating;
    memcpy(&floating, &integer, sizeof(floating));
    return floating;
  }

  /**
   * @brief Checks whether the bit-casted floating-point value is +/-0
   *
   * @param integer_rep The bit-casted floating value to check if is +/-0
   * @return True if is a zero, else false
   */
  CUDF_HOST_DEVICE inline static bool is_zero(IntegralType integer_rep)
  {
    // It's a zero if every non-sign bit is zero
    return ((integer_rep & ~sign_mask) == 0);
  }

  /**
   * @brief Extracts the sign bit of a bit-casted floating-point number
   *
   * @param integer_rep The bit-casted floating value to extract the exponent from
   * @return The sign bit
   */
  CUDF_HOST_DEVICE inline static bool get_is_negative(IntegralType integer_rep)
  {
    // Extract the sign bit:
    return static_cast<bool>(sign_mask & integer_rep);
  }

  /**
   * @brief Extracts the significand and exponent of a bit-casted floating-point number
   *
   * @note This returns (1 - exponent_bias) for denormals. Zeros/inf/NaN not handled.
   *
   * @param integer_rep The bit-casted floating value to extract the exponent from
   * @return The stored base-2 exponent, or (1 - exponent_bias) for denormals
   */
  CUDF_HOST_DEVICE inline static std::pair<IntegralType, int> get_significand_and_exp2(
    IntegralType integer_rep)
  {
    // Extract the significand
    auto significand = (integer_rep & mantissa_mask);

    // Extract the exponent bits.
    auto const exponent_bits = integer_rep & exponent_mask;

    // Notes on special values of exponent_bits:
    // bits = exponent_mask is +/-inf or NaN, but those are handled prior to input.
    // bits = 0 is either a denormal (handled below) or a zero (handled earlier by caller).
    int floating_exp2;
    if (exponent_bits == 0) {
      // Denormal values are 2^(1 - exponent_bias) * Sum_i(B_i * 2^-i)
      // Where i is the i-th mantissa bit (counting from the LEFT, starting at 1),
      // and B_i is the value of that bit (0 or 1)
      // So e.g. for the minimum denormal, only the lowest bit is set:
      // FLT_TRUE_MIN = 2^(1 - 127) * 2^-23 = 2^-149
      // DBL_TRUE_MIN = 2^(1 - 1023) * 2^-52 = 2^-1074
      floating_exp2 = 1 - exponent_bias;
    } else {
      // Extract the exponent value: shift the bits down and subtract the bias.
      auto const shifted_exponent_bits = exponent_bits >> num_mantissa_bits;
      floating_exp2                    = static_cast<int>(shifted_exponent_bits) - exponent_bias;

      // Set the high bit for the understood 1/2
      significand |= understood_bit_mask;
    }

    // To convert the mantissa to an integer, we effectively applied #-mantissa-bits
    // powers of 2 to convert the fractional value to an integer, so subtract them off here
    int const exp2 = floating_exp2 - num_mantissa_bits;

    return {significand, exp2};
  }

  /**
   * @brief Sets the sign bit of a floating-point number
   *
   * @param floating The floating-point value to set the sign of. Must be positive.
   * @param is_negative The sign bit to set for the floating-point number
   * @return The input floating-point value with the chosen sign
   */
  CUDF_HOST_DEVICE inline static FloatingType set_is_negative(FloatingType floating,
                                                              bool is_negative)
  {
    // Convert floating to integer
    IntegralType integer_rep = bit_cast_to_integer(floating);

    // Set the sign bit. Note that the input floating-point number must be positive (bit = 0).
    integer_rep |= (IntegralType(is_negative) << sign_bit_index);

    // Convert back to float
    return bit_cast_to_floating(integer_rep);
  }

  /**
   * @brief Adds to the base-2 exponent of a floating-point number
   *
   * @note Where called, the input is guaranteed to be a positive whole number.
   *
   * @param floating The floating value to add to the exponent of. Must be positive.
   * @param exp2 The power-of-2 to add to the floating-point number
   * @return The input floating-point value * 2^exp2
   */
  CUDF_HOST_DEVICE inline static FloatingType add_exp2(FloatingType floating, int exp2)
  {
    // Note that the input floating-point number is positive (& whole), so we don't have to
    // worry about the sign here; the sign will be set later in set_is_negative()

    // Convert floating to integer
    auto integer_rep = bit_cast_to_integer(floating);

    // Extract the currently stored (biased) exponent
    using SignedType   = std::make_signed_t<IntegralType>;
    auto exponent_bits = integer_rep & exponent_mask;
    auto stored_exp2   = static_cast<SignedType>(exponent_bits >> num_mantissa_bits);

    // Add the additional power-of-2
    stored_exp2 += exp2;

    // Check for exponent over/under-flow.
    if (stored_exp2 <= 0) {
      // Denormal (zero handled prior to input)

      // Early out if bit shift will zero it anyway.
      // Note: We must handle this explicitly, as too-large a bit-shift is UB
      auto const bit_shift = -stored_exp2 + 1;  //+1 due to understood bit set below
      if (bit_shift > num_mantissa_bits) { return 0.0; }

      // Clear the exponent bits (zero means 2^-126/2^-1022 w/ no understood bit)
      integer_rep &= (~exponent_mask);

      // The input floating-point number has an "understood" bit that we need to set
      // prior to bit-shifting. Set the understood bit.
      integer_rep |= understood_bit_mask;

      // Convert to denormal: bit shift off the low bits
      integer_rep >>= bit_shift;
    } else if (stored_exp2 >= static_cast<SignedType>(unshifted_exponent_mask)) {
      // Overflow: Set infinity
      return cuda::std::numeric_limits<FloatingType>::infinity();
    } else {
      // Normal number: Clear existing exponent bits and set new ones
      exponent_bits = static_cast<IntegralType>(stored_exp2) << num_mantissa_bits;
      integer_rep &= (~exponent_mask);
      integer_rep |= exponent_bits;
    }

    // Convert back to float
    return bit_cast_to_floating(integer_rep);
  }
};

/**
 * @brief Determine the number of significant bits in an integer
 *
 * @tparam T Type of input integer value. Must be either uint32_t, uint64_t, or __uint128_t
 * @param value The integer whose bits are being counted
 * @return The number of significant bits: the # of bits - # of leading zeroes
 */
template <typename T,
          CUDF_ENABLE_IF(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t> ||
                         std::is_same_v<T, __uint128_t>)>
CUDF_HOST_DEVICE inline int count_significant_bits(T value)
{
#ifdef __CUDA_ARCH__
  if constexpr (std::is_same_v<T, uint64_t>) {
    return 64 - __clzll(static_cast<int64_t>(value));
  } else if constexpr (std::is_same_v<T, uint32_t>) {
    return 32 - __clz(static_cast<int32_t>(value));
  } else if constexpr (std::is_same_v<T, __uint128_t>) {
    // 128 bit type, must break up into high and low components
    auto const high_bits = static_cast<int64_t>(value >> 64);
    auto const low_bits  = static_cast<int64_t>(value);
    return 128 - (__clzll(high_bits) + static_cast<int>(high_bits == 0) * __clzll(low_bits));
  }
#else
  // Undefined behavior to call __builtin_clzll() with zero in gcc and clang
  if (value == 0) { return 0; }

  if constexpr (std::is_same_v<T, uint64_t>) {
    return 64 - __builtin_clzll(value);
  } else if constexpr (std::is_same_v<T, uint32_t>) {
    return 32 - __builtin_clz(value);
  } else if constexpr (std::is_same_v<T, __uint128_t>) {
    // 128 bit type, must break up into high and low components
    auto const high_bits = static_cast<uint64_t>(value >> 64);
    if (high_bits == 0) {
      return 64 - __builtin_clzll(static_cast<uint64_t>(value));
    } else {
      return 128 - __builtin_clzll(high_bits);
    }
  }
#endif
}

/**
 * @brief Recursively calculate a signed large power of 10 (>= 10^19) that can only be stored in an
 * 128bit integer
 *
 * @note Intended to be run at compile time.
 *
 * @tparam Exp10 The power of 10 to calculate
 * @return Returns 10^Exp10
 */
template <int Exp10>
constexpr __uint128_t large_power_of_10()
{
  // Stop at 10^19 to speed up compilation; literals can be used for smaller powers of 10.
  static_assert(Exp10 >= 19);
  if constexpr (Exp10 == 19)
    return __uint128_t(10000000000000000000ULL);
  else
    return large_power_of_10<Exp10 - 1>() * __uint128_t(10);
}

/**
 * @brief Divide by a power of 10 that fits within a 32bit integer.
 *
 * @tparam T Type of value to be divided-from.
 * @param value The number to be divided-from.
 * @param exp10 The power-of-10 of the denominator, from 0 to 9 inclusive.
 * @return Returns value / 10^exp10
 */
template <typename T, typename cuda::std::enable_if_t<cuda::std::is_unsigned_v<T>>* = nullptr>
CUDF_HOST_DEVICE inline T divide_power10_32bit(T value, int exp10)
{
  // Computing division this way is much faster than the alternatives.
  // Division is not implemented in GPU hardware, and the compiler will often implement it as a
  // multiplication of the reciprocal of the denominator, requiring a conversion to floating point.
  // Ths is especially slow for larger divides that have to use the FP64 pipeline, where threads
  // bottleneck.

  // Instead, if the compiler can see exactly what number it is dividing by, it can
  // produce much more optimal assembly, doing bit shifting, multiplies by a constant, etc.
  // For the compiler to see the value though, array lookup (with exp10 as the index)
  // is not sufficient: We have to use a switch statement. Although this introduces a branch,
  // it is still much faster than doing the divide any other way.
  // Perhaps an array can be used in C++23 with the assume attribute?

  // Since we're optimizing division this way, we have to do this for multiplication as well.
  // That's because doing them in different ways (switch, array, runtime-computation, etc.)
  // increases the register pressure on all kernels that use fixed_point types, specifically slowing
  // down some of the PYMOD and join benchmarks.

  // This is split up into separate functions for 32-, 64-, and 128-bit denominators.
  // That way we limit the templated, inlined code generation to the exponents that are
  // capable of being represented. Combining them together into a single function again
  // introduces too much pressure on the kernels that use this code, slowing down their benchmarks.
  // It also dramatically slows down the compile time.

  switch (exp10) {
    case 0: return value;
    case 1: return value / 10U;
    case 2: return value / 100U;
    case 3: return value / 1000U;
    case 4: return value / 10000U;
    case 5: return value / 100000U;
    case 6: return value / 1000000U;
    case 7: return value / 10000000U;
    case 8: return value / 100000000U;
    case 9: return value / 1000000000U;
    default: return 0;
  }
}

/**
 * @brief Divide by a power of 10 that fits within a 64bit integer.
 *
 * @tparam T Type of value to be divided-from.
 * @param value The number to be divided-from.
 * @param exp10 The power-of-10 of the denominator, from 0 to 19 inclusive.
 * @return Returns value / 10^exp10
 */
template <typename T, typename cuda::std::enable_if_t<cuda::std::is_unsigned_v<T>>* = nullptr>
CUDF_HOST_DEVICE inline T divide_power10_64bit(T value, int exp10)
{
  // See comments in divide_power10_32bit() for discussion.
  switch (exp10) {
    case 0: return value;
    case 1: return value / 10U;
    case 2: return value / 100U;
    case 3: return value / 1000U;
    case 4: return value / 10000U;
    case 5: return value / 100000U;
    case 6: return value / 1000000U;
    case 7: return value / 10000000U;
    case 8: return value / 100000000U;
    case 9: return value / 1000000000U;
    case 10: return value / 10000000000ULL;
    case 11: return value / 100000000000ULL;
    case 12: return value / 1000000000000ULL;
    case 13: return value / 10000000000000ULL;
    case 14: return value / 100000000000000ULL;
    case 15: return value / 1000000000000000ULL;
    case 16: return value / 10000000000000000ULL;
    case 17: return value / 100000000000000000ULL;
    case 18: return value / 1000000000000000000ULL;
    case 19: return value / 10000000000000000000ULL;
    default: return 0;
  }
}

/**
 * @brief Divide by a power of 10 that fits within a 128bit integer.
 *
 * @tparam T Type of value to be divided-from.
 * @param value The number to be divided-from.
 * @param exp10 The power-of-10 of the denominator, from 0 to 38 inclusive.
 * @return Returns value / 10^exp10.
 */
template <typename T, typename cuda::std::enable_if_t<cuda::std::is_unsigned_v<T>>* = nullptr>
CUDF_HOST_DEVICE inline constexpr T divide_power10_128bit(T value, int exp10)
{
  // See comments in divide_power10_32bit() for an introduction.
  switch (exp10) {
    case 0: return value;
    case 1: return value / 10U;
    case 2: return value / 100U;
    case 3: return value / 1000U;
    case 4: return value / 10000U;
    case 5: return value / 100000U;
    case 6: return value / 1000000U;
    case 7: return value / 10000000U;
    case 8: return value / 100000000U;
    case 9: return value / 1000000000U;
    case 10: return value / 10000000000ULL;
    case 11: return value / 100000000000ULL;
    case 12: return value / 1000000000000ULL;
    case 13: return value / 10000000000000ULL;
    case 14: return value / 100000000000000ULL;
    case 15: return value / 1000000000000000ULL;
    case 16: return value / 10000000000000000ULL;
    case 17: return value / 100000000000000000ULL;
    case 18: return value / 1000000000000000000ULL;
    case 19: return value / 10000000000000000000ULL;
    case 20: return value / large_power_of_10<20>();
    case 21: return value / large_power_of_10<21>();
    case 22: return value / large_power_of_10<22>();
    case 23: return value / large_power_of_10<23>();
    case 24: return value / large_power_of_10<24>();
    case 25: return value / large_power_of_10<25>();
    case 26: return value / large_power_of_10<26>();
    case 27: return value / large_power_of_10<27>();
    case 28: return value / large_power_of_10<28>();
    case 29: return value / large_power_of_10<29>();
    case 30: return value / large_power_of_10<30>();
    case 31: return value / large_power_of_10<31>();
    case 32: return value / large_power_of_10<32>();
    case 33: return value / large_power_of_10<33>();
    case 34: return value / large_power_of_10<34>();
    case 35: return value / large_power_of_10<35>();
    case 36: return value / large_power_of_10<36>();
    case 37: return value / large_power_of_10<37>();
    case 38: return value / large_power_of_10<38>();
    default: return 0;
  }
}

/**
 * @brief Multiply by a power of 10 that fits within a 32bit integer.
 *
 * @tparam T Type of value to be multiplied.
 * @param value The number to be multiplied.
 * @param exp10 The power-of-10 of the multiplier, from 0 to 9 inclusive.
 * @return Returns value * 10^exp10
 */
template <typename T, typename cuda::std::enable_if_t<cuda::std::is_unsigned_v<T>>* = nullptr>
CUDF_HOST_DEVICE inline constexpr T multiply_power10_32bit(T value, int exp10)
{
  // See comments in divide_power10_32bit() for discussion.
  switch (exp10) {
    case 0: return value;
    case 1: return value * 10U;
    case 2: return value * 100U;
    case 3: return value * 1000U;
    case 4: return value * 10000U;
    case 5: return value * 100000U;
    case 6: return value * 1000000U;
    case 7: return value * 10000000U;
    case 8: return value * 100000000U;
    case 9: return value * 1000000000U;
    default: return 0;
  }
}

/**
 * @brief Multiply by a power of 10 that fits within a 64bit integer.
 *
 * @tparam T Type of value to be multiplied.
 * @param value The number to be multiplied.
 * @param exp10 The power-of-10 of the multiplier, from 0 to 19 inclusive.
 * @return Returns value * 10^exp10
 */
template <typename T, typename cuda::std::enable_if_t<cuda::std::is_unsigned_v<T>>* = nullptr>
CUDF_HOST_DEVICE inline constexpr T multiply_power10_64bit(T value, int exp10)
{
  // See comments in divide_power10_32bit() for discussion.
  switch (exp10) {
    case 0: return value;
    case 1: return value * 10U;
    case 2: return value * 100U;
    case 3: return value * 1000U;
    case 4: return value * 10000U;
    case 5: return value * 100000U;
    case 6: return value * 1000000U;
    case 7: return value * 10000000U;
    case 8: return value * 100000000U;
    case 9: return value * 1000000000U;
    case 10: return value * 10000000000ULL;
    case 11: return value * 100000000000ULL;
    case 12: return value * 1000000000000ULL;
    case 13: return value * 10000000000000ULL;
    case 14: return value * 100000000000000ULL;
    case 15: return value * 1000000000000000ULL;
    case 16: return value * 10000000000000000ULL;
    case 17: return value * 100000000000000000ULL;
    case 18: return value * 1000000000000000000ULL;
    case 19: return value * 10000000000000000000ULL;
    default: return 0;
  }
}

/**
 * @brief Multiply by a power of 10 that fits within a 128bit integer.
 *
 * @tparam T Type of value to be multiplied.
 * @param value The number to be multiplied.
 * @param exp10 The power-of-10 of the multiplier, from 0 to 38 inclusive.
 * @return Returns value * 10^exp10.
 */
template <typename T, typename cuda::std::enable_if_t<cuda::std::is_unsigned_v<T>>* = nullptr>
CUDF_HOST_DEVICE inline constexpr T multiply_power10_128bit(T value, int exp10)
{
  // See comments in divide_power10_128bit() for discussion.
  switch (exp10) {
    case 0: return value;
    case 1: return value * 10U;
    case 2: return value * 100U;
    case 3: return value * 1000U;
    case 4: return value * 10000U;
    case 5: return value * 100000U;
    case 6: return value * 1000000U;
    case 7: return value * 10000000U;
    case 8: return value * 100000000U;
    case 9: return value * 1000000000U;
    case 10: return value * 10000000000ULL;
    case 11: return value * 100000000000ULL;
    case 12: return value * 1000000000000ULL;
    case 13: return value * 10000000000000ULL;
    case 14: return value * 100000000000000ULL;
    case 15: return value * 1000000000000000ULL;
    case 16: return value * 10000000000000000ULL;
    case 17: return value * 100000000000000000ULL;
    case 18: return value * 1000000000000000000ULL;
    case 19: return value * 10000000000000000000ULL;
    case 20: return value * large_power_of_10<20>();
    case 21: return value * large_power_of_10<21>();
    case 22: return value * large_power_of_10<22>();
    case 23: return value * large_power_of_10<23>();
    case 24: return value * large_power_of_10<24>();
    case 25: return value * large_power_of_10<25>();
    case 26: return value * large_power_of_10<26>();
    case 27: return value * large_power_of_10<27>();
    case 28: return value * large_power_of_10<28>();
    case 29: return value * large_power_of_10<29>();
    case 30: return value * large_power_of_10<30>();
    case 31: return value * large_power_of_10<31>();
    case 32: return value * large_power_of_10<32>();
    case 33: return value * large_power_of_10<33>();
    case 34: return value * large_power_of_10<34>();
    case 35: return value * large_power_of_10<35>();
    case 36: return value * large_power_of_10<36>();
    case 37: return value * large_power_of_10<37>();
    case 38: return value * large_power_of_10<38>();
    default: return 0;
  }
}

/**
 * @brief Multiply an integer by a power of 10.
 *
 * @note Use this function if you have no a-priori knowledge of what exp10 might be.
 * If you do, prefer calling the bit-size-specific versions
 *
 * @tparam Rep Representation type needed for integer exponentiation
 * @tparam T Integral type of value to be multiplied.
 * @param value The number to be multiplied.
 * @param exp10 The power-of-10 of the multiplier.
 * @return Returns value * 10^exp10
 */
template <typename Rep,
          typename T,
          typename cuda::std::enable_if_t<(cuda::std::is_unsigned_v<T>)>* = nullptr>
CUDF_HOST_DEVICE inline constexpr T multiply_power10(T value, int exp10)
{
  // Use this function if you have no knowledge of what exp10 might be
  // If you do, prefer calling the bit-size-specific versions
  if constexpr (sizeof(Rep) <= 4) {
    return multiply_power10_32bit(value, exp10);
  } else if constexpr (sizeof(Rep) <= 8) {
    return multiply_power10_64bit(value, exp10);
  } else {
    return multiply_power10_128bit(value, exp10);
  }
}

/**
 * @brief Divide an integer by a power of 10.
 *
 * @note Use this function if you have no a-priori knowledge of what exp10 might be.
 * If you do, prefer calling the bit-size-specific versions
 *
 * @tparam Rep Representation type needed for integer exponentiation
 * @tparam T Integral type of value to be divided-from.
 * @param value The number to be divided-from.
 * @param exp10 The power-of-10 of the denominator.
 * @return Returns value / 10^exp10
 */
template <typename Rep,
          typename T,
          typename cuda::std::enable_if_t<(cuda::std::is_unsigned_v<T>)>* = nullptr>
CUDF_HOST_DEVICE inline constexpr T divide_power10(T value, int exp10)
{
  // Use this function if you have no knowledge of what exp10 might be
  // If you do, prefer calling the bit-size-specific versions
  if constexpr (sizeof(Rep) <= 4) {
    return divide_power10_32bit(value, exp10);
  } else if constexpr (sizeof(Rep) <= 8) {
    return divide_power10_64bit(value, exp10);
  } else {
    return divide_power10_128bit(value, exp10);
  }
}

/**
 * @brief Perform a bit-shift left, guarding against undefined behavior
 *
 * @tparam IntegerType Type of input unsigned integer value
 * @param value The integer whose bits are being shifted
 * @param bit_shift The number of bits to shift left
 * @return The bit-shifted integer, except max value if overflow would occur
 */
template <typename IntegerType, CUDF_ENABLE_IF(cuda::std::is_unsigned_v<IntegerType>)>
CUDF_HOST_DEVICE inline IntegerType guarded_left_shift(IntegerType value, int bit_shift)
{
  // Bit shifts larger than this are undefined behavior
  static constexpr int max_safe_bit_shift = cuda::std::numeric_limits<IntegerType>::digits - 1;
  return (bit_shift <= max_safe_bit_shift) ? value << bit_shift
                                           : cuda::std::numeric_limits<IntegerType>::max();
}

/**
 * @brief Perform a bit-shift right, guarding against undefined behavior
 *
 * @tparam IntegerType Type of input unsigned integer value
 * @param value The integer whose bits are being shifted
 * @param bit_shift The number of bits to shift right
 * @return The bit-shifted integer, which is zero on underflow
 */
template <typename IntegerType, CUDF_ENABLE_IF(cuda::std::is_unsigned_v<IntegerType>)>
CUDF_HOST_DEVICE inline IntegerType guarded_right_shift(IntegerType value, int bit_shift)
{
  // Bit shifts larger than this are undefined behavior
  static constexpr int max_safe_bit_shift = cuda::std::numeric_limits<IntegerType>::digits - 1;
  return (bit_shift <= max_safe_bit_shift) ? value >> bit_shift : 0;
}

/**
 * @brief Helper struct with common constants needed by the floating <--> decimal conversions
 */
template <typename FloatingType>
struct shifting_constants {
  /// Whether the type is double
  static constexpr bool is_double = cuda::std::is_same_v<FloatingType, double>;

  /// Integer type that can hold the value of the significand
  using IntegerRep = std::conditional_t<is_double, uint64_t, uint32_t>;

  /// Num bits needed to hold the significand
  static constexpr auto num_significand_bits = cuda::std::numeric_limits<FloatingType>::digits;

  /// Shift data back and forth in space of a type with 2x the starting bits, to give us enough room
  using ShiftingRep = std::conditional_t<is_double, __uint128_t, uint64_t>;

  // The significand of a float / double is 24 / 53 bits
  // However, to uniquely represent each double / float as different #'s in decimal
  // you need 17 / 9 digits (from std::numeric_limits<T>::max_digits10)
  // To represent 10^17 / 10^9, you need 57 / 30 bits
  // +1 for adding half of a bit to handle the 1.2 -> 1.19999 conversion problem
  // So we need to keep track of this # of bits during shifting to ensure no info is lost
  /// # bits needed to represent the value
  static constexpr int num_rep_bits = is_double ? 58 : 31; //57 : 30;

  // We will be alternately shifting our data back and forth by powers of 2 and 10 to convert
  // between floating and decimal (see shifting functions for details).

  // To iteratively shift back and forth, our 2's (bit-) and 10's (divide-/multiply-) shifts must
  // be of nearly the same magnitude, or else we'll over-/under-flow our shifting integer

  // 2^10 is approximately 10^3, so the largest shifts will have a 10/3 ratio
  // The difference between 2^10 and 10^3 is 1024/1000: 2.4%
  // So every time we shift by 10 bits and 3 decimal places, the 2s shift is an extra 2.4%

  // This 2.4% error compounds each time we do an iteration.
  // The min (normal) float is 2^-126.
  // Min denormal: 2^-126 * 2^-23 (mantissa bits): 2^-149 = ~1.4E-45
  // With our 10/3 shifting ratio, 149 (bit-shifts) * (3 / 10) = 44.7 (10s-shifts)
  // 10^(-44.7) = 2E-45, which is off by ~1.4x from 1.4E-45

  // Similarly, the min (normal) double is 2^-1022.
  // Min denormal: 2^-1022 * 2^-52 (mantissa bits): 2^-1074 = 4.94E-324
  // With our 10/3 shifting ratio, 1074 (bit-shifts) * (3 / 10) = 322.2 (10s-shifts)
  // 10^(-322.2) = 6.4E-323, which is off by ~13.2x from 4.94E-324

  // To account for this compounding error, we can either complicate our loop code (slow),
  // or use extra bits (in the direction we're shifting the 2s!) to compensate:
  // 4 extra bits for doubles (2^4 = 16 > 13.2x error), 1 extra for floats (2 > 1.4x error)
  /// # buffer bits to account for shifting error
  static constexpr int num_2s_shift_buffer_bits = is_double ? 4 : 1;

  // How much room do we have for shifting?
  // Float: 64-bit ShiftingRep - 31 (rep + buffer) = 33 bits. 2^33 = 8.6E9
  // Double: 128-bit ShiftingRep - 61 (rep + buffer) = 67 bits. 2^67 = 1.5E20
  // Thus for double / float we can shift up to 20 / 9 decimal places at once

  // But, we need to stick to our 10-bits / 3-decimals shift ratio to not over/under-flow.
  // To simplify our loop code, we'll keep to this ratio by instead shifting a max of
  // 18 / 9 decimal places, for double / float (60 / 30 bits)
  /// Max at-once decimal place shift
  static constexpr int max_digits_shift = is_double ? 18 : 9;
  /// Max at-once bit shift
  static constexpr int max_bits_shift = max_digits_shift * 10 / 3;

  // Pre-calculate 10^max_digits_shift. Note that 10^18 / 10^9 fits within IntegerRep
  /// 10^max_digits_shift
  static constexpr auto max_digits_shift_pow =
    multiply_power10<IntegerRep>(IntegerRep(1), max_digits_shift);
};

template <typename T, typename cuda::std::enable_if_t<cuda::std::is_integral_v<T>>* = nullptr>
CUDF_HOST_DEVICE cuda::std::pair<T, int> add_half_bit(T integral_mantissa, int exp2, int exp10)
{
  // The user-supplied scale may truncate information, so we need to talk about rounding.
  // We have chosen not to round, so we want 1.23456f with scale -4 to be decimal 12345

  // But if we don't round at all, 1.2 (double) with scale -1 is 11 instead of 12!
  // Why? Because 1.2 (double) is actually stored as 1.1999999... which we truncate to 1.1
  // While correct (given our choice to truncate), this is surprising and undesirable.
  // This problem happens because 1.2 is not perfectly representable in floating point,
  // and the value 1.199999... happened to be closer to 1.2 than the next value (1.2000...1...)

  // If the scale truncates >= 1 bit (we didn't choose to keep exactly 1.1999...), how
  // do we make sure we store 1.2? All we have to do is add 1 ulp! (unit in the last place)
  // Then 1.1999... becomes 1.2000...1... which truncates to 1.2.
  // And if it was 1.2000...1... adding 1 ulp still truncates to 1.2: the result is unchanged.

  // If we don't truncate any bits, don't add anything, else we get the wrong result. 
  // So when does the user-supplied scale truncate info?

//Add 0 Always: 1.2 -> 1.1999 -> 1.1: BAD
//Add 0 on no truncate: 1.2 -> 1.1999: BAD
//Add 1 Always: 10.03 -> 10.030...02...: BAD
//Add 1 on truncate Any: 97938.20 -> 97938.21: BAD
//Add 0.5 Always: 10.03 -> 10.030...02...: BAD

//Guaranteed ok:
//Add 1 on truncate >= 1 bit
//Add 0.5

//LATEST
  //this shift is zero for normals, & > 0 for denormals
  //shift out to normal lineup spot, then add 1 as normal

  using FloatingType = std::conditional_t<std::is_same_v<T, uint32_t>, float, double>;
  using Constants = shifting_constants<FloatingType>;
  auto const lineup_shift = Constants::num_significand_bits - count_significant_bits(integral_mantissa) + 1;

  integral_mantissa <<= lineup_shift;
  exp2 -= lineup_shift;
  integral_mantissa++; //add half of a bit
  return {integral_mantissa, exp2};
}

/**
 * @brief Increment integer rep of floating point if conversion causes truncation
 *
 * @note This fixes problems like 1.2 (value = 1.1999...) at scale -1 -> 11
 *
 * @tparam T Type of integer holding the floating-point significand
 * @param integral_mantissa The integer representation of the floating-point significand
 * @param exp2 The power of 2 that needs to be applied to the significand
 * @param exp10 The power of 10 that needs to be applied to the significand
 * @return significand, incremented if the conversion to decimal causes truncation
 */
/*
template <typename T, typename cuda::std::enable_if_t<cuda::std::is_integral_v<T>>* = nullptr>
CUDF_HOST_DEVICE T increment_on_truncation(T const integral_mantissa,
                                           int const exp2,
                                           int const exp10)
{



  // The only way that this produces the incorrect result is if, when we entered 1.19999...,
  // we truly meant 1.19999... (exactly, out to the very last digit), but then decided to truncate
  // anyway. By choosing to truncate, you're saying you don't actually care about that level of
  // precision, so being off by 1 ulp should be just fine, compared to screwing up 1.2

//The conversion from 1.2 -> 1.1999... will only ever be off by <= 0.5 bits, 
//Oh my god. What if we just shifted left by 1, ++exp2, and THEN added 1.
//Well, we are running out of shifting space as it is ...
//(else the other # would have been chosen!)
//So if we truncate MORE than half of a bit, we can safely ++:
//e.g. if we were low by 0.4 bits, we ++ to 0.6 bits and floor to same value
//but if we were high by 0.1 bits, we ++ 1.1 bits and 


  // So when does the user-supplied scale truncate info?
  // integral_mantissa is some integer. Only 7 decimal digits of which are truly precise. 
  // We apply exp2 to get its true value, and exp10 to set the decimal place
  // If exp2 > 0 (left-shift), truncate if divide by more 10s (exp10 > 0) than we shift up by 2s
  // If exp2 < 0 (right-shift), truncate if multiply by LESS OR SAME 10s (exp10 < 0) than we shift down by 2s

  // Float is precise to 7 decimal places. If the user chose a scale for 7 decimal places,  
  // then the conversion does not truncate. 
//  int const corresponding_exp10 = 90 * exp2 / 299;
  // 1.2 scale -1 = 12 * 10^-1
  // 1200 scale 1 = 120 * 10^1
  // If exp10 > 0, truncate if divide by more 10s than we shift up by 2s
//  bool const conversion_truncates_digit = 
////    (-exp10 <= -corresponding_exp10)
//    (exp10 > corresponding_exp10) || ((exp10 == corresponding_exp10) && (exp10 < 0));
(-2 > -2.1) -> (2 < 2.1): truncates 0.107 digits

//If you truncate 0.1 bits, and choose NOT to add 1, when is that the WRONG thing?
//if 1.2 -> 1.1999....991, 

//If we are truncating by a full bit, can definitely ++ safely
//If not truncating, definitely cannot ++
//If truncating by part of a bit ... not safe, choosing between one error or another. 


  // For powers > 0: When the 10s (scale) shift is larger than the corresponding bit-shift.
  // For powers < 0: When the 10s shift is less than the corresponding bit-shift.

  // Corresponding bit-shift:
  // 2^10 is approximately 10^3, but this is off by 1.024%
  // 1.024^30 is 2.03704, so this is high by one bit for every 30*3 = 90 powers of 10
  // So 10^N = 2^(10*N/3 - N/90) = 2^(299*N/90)
  int const corresponding_exp2 = 299 * exp10 / 90;

  // If exp10 > 0, truncate if divide by more 10s than we shift up by 2s
  // If exp10 < 0, truncate if shift down by more OR THE SAME 2s than multiply by 10s
  // Truncate on the same: because for our approximation 2^299 > 10^90
  // Note that this works for both +/- exponents
  bool const conversion_truncates =
    (exp2 < corresponding_exp2) || ((exp2 == corresponding_exp2) && (exp2 < 0));
(-7 < -6.6444444) it truncates less than half a bit

  // (Potentially) increment and return
  return integral_mantissa + static_cast<T>(conversion_truncates);
}
*/

/**
 * @brief Perform lossless base-2 -> base-10 fixed-point conversion for exp10 > 0
 *
 * @note Info is lost if the chosen scale factor truncates information.
 *
 * @tparam FloatingType The type of the original floating-point value we are converting from
 * @param base2_value The base-2 fixed-point value we are converting from
 * @param exp2 The number of powers of 2 to apply to convert from base-2
 * @param exp10 The number of powers of 10 to apply to reach the desired scale factor
 * @return Magnitude of the converted-to decimal integer
 */

template <typename FloatingType,
          typename cuda::std::enable_if_t<cuda::std::is_floating_point_v<FloatingType>>* = nullptr>
CUDF_HOST_DEVICE inline typename shifting_constants<FloatingType>::ShiftingRep
shift_to_decimal_posexp(typename shifting_constants<FloatingType>::IntegerRep const base2_value,
                        int exp2,
                        int exp10)
{
  // To convert to decimal, we need to apply the input powers of 2 and 10
  // The result will be (integer) base2_value * (2^exp2) / (10^exp10)
  // Output type is ShiftingRep

  // Here exp10 > 0 and exp2 > 0, so we need to shift left by 2s and divide by 10s.
  // To do this losslessly, we will iterate back and forth between them, shifting
  // up by 2s and down by 10s until all of the powers have been applied.

  // However the input base2_value type has virtually no spare room to shift our data
  // without over- or under-flowing and losing precision.
  // So we'll cast up to ShiftingRep: uint64 for float's, __uint128_t for double's
  using Constants   = shifting_constants<FloatingType>;
  using ShiftingRep = typename Constants::ShiftingRep;
  auto shifting_rep = static_cast<ShiftingRep>(base2_value);

  // We want to start with our significand bits at the top of the num_rep_bits range, 
  // so that we don't lose information we need on intermediary right-shifts.
  // For all normal numbers this bit shift is a fixed distance, due to the understood 2^0 bit.
  // Note we also need to shift one less, due to adding half of a bit earlier. 
//  static constexpr int normal_lineup_shift = Constants::num_rep_bits - Constants::num_significand_bits - 1;

  // We want to start by lining up our bits in num_rep_bits (see comments on normal_lineup_shift),
  // but since we start by bit-shifting up anyway, combine the normal_lineup_shift & max_bits_shift.
  // Note that since we're shifting 2s up, we need num_2s_shift_buffer_bits space on the high side,
  // which we do (our max bit shift is low enough that we don't shift into the highest bits)
//  static constexpr int max_init_shift = normal_lineup_shift + Constants::max_bits_shift;
static constexpr int shift_up_to = sizeof(ShiftingRep) * 8 - Constants::num_2s_shift_buffer_bits;
static constexpr int shift_from = Constants::num_significand_bits + 1;
static constexpr int max_init_shift = shift_up_to - shift_from;

  // If our total bit shift is less than this, we don't need to iterate
  if (exp2 <= max_init_shift) {
    // Shift bits left, divide by 10s to apply the scale factor, and we're done.
    return divide_power10<ShiftingRep>(shifting_rep << exp2, exp10);
  }

  // We need to iterate. Do the combined initial shift
  shifting_rep <<= max_init_shift;
  exp2 -= max_init_shift;

  // Iterate, dividing by 10s and shifting up by 2s until we're almost done
  while (exp10 > Constants::max_digits_shift) {
    // More decimal places to shift than we have room: Divide the max number of 10s

    // Note that the result of this division is guaranteed to fit within the low half of the bits.
    // The highest set bit is num_rep_bits + num_2s_shift_buffer_bits + max_bits_shift
    // For float this is 30 + 1 + 30 = 61, for double 57 + 4 + 60 = 121
    // 2^61 / 10^9 (~2^30) is ~2^31, and 2^121 / 10^18 (~2^60) is ~2^61
    // As a future optimization, we could use a faster division routine that takes this account.
    shifting_rep /= Constants::max_digits_shift_pow;
    exp10 -= Constants::max_digits_shift;

    // If our remaining bit shift is less than the max, we're finished iterating
    if (exp2 <= Constants::max_bits_shift) {
      // Shift bits left, divide by 10s to apply the scale factor, and we're done.
      // Note: This divide result may not fit in the low half of the bit range
      return divide_power10<ShiftingRep>(shifting_rep << exp2, exp10);
    }

    // Shift the max number of bits left again
    shifting_rep <<= Constants::max_bits_shift;
    exp2 -= Constants::max_bits_shift;
  }

  // Last 10s-shift: Divdie all remaining decimal places, shift all remaining bits, then bail
  // Note: This divide result may not fit in the low half of the bit range
  // But the divisor is less than the max-shift, and thus fits within 64 / 32 bits
  if constexpr (Constants::is_double) {
    shifting_rep = divide_power10_64bit(shifting_rep, exp10);
  } else {
    shifting_rep = divide_power10_32bit(shifting_rep, exp10);
  }

  // Final bit shift: Shift may be large, guard against UB
  // NOTE: This can overflow!
  return guarded_left_shift(shifting_rep, exp2);
}

/**
 * @brief Perform lossless base-2 -> base-10 fixed-point conversion for exp10 < 0
 *
 * @note Info is lost if the chosen scale factor truncates information.
 *
 * @tparam FloatingType The type of the original floating-point value we are converting from
 * @param base2_value The base-2 fixed-point value we are converting from
 * @param exp2 The number of powers of 2 to apply to convert from base-2
 * @param exp10 The number of powers of 10 to apply to reach the desired scale factor
 * @return Magnitude of the converted-to decimal integer
 */
template <typename Rep,
          typename FloatingType,
          typename cuda::std::enable_if_t<cuda::std::is_floating_point_v<FloatingType>>* = nullptr>
CUDF_HOST_DEVICE inline typename shifting_constants<FloatingType>::ShiftingRep
shift_to_decimal_negexp(typename shifting_constants<FloatingType>::IntegerRep base2_value,
                        int exp2,
                        int exp10)
{
  // This is similar to shift_to_decimal_posexp(), except exp10 < 0 & exp2 < 0
  // See comments in that function for details.
  // Instead here we need to multiply by 10s and shift right by 2s

  // ShiftingRep: uint64 for float's, __uint128_t for double's
  using Constants   = shifting_constants<FloatingType>;
  using ShiftingRep = typename Constants::ShiftingRep;
  auto shifting_rep = static_cast<ShiftingRep>(base2_value);

  // Convert to using positive values so we don't have keep negating
  int exp10_mag = -exp10;
  int exp2_mag = -exp2;

  // For performing final 10s-shift
  auto final_shifts_low10s = [&]() {
    // Last 10s-shift: multiply all remaining decimal places, shift all remaining bits, then bail
    // The multiplier is less than the max-shift, and thus fits within 64 / 32 bits
    if constexpr (Constants::is_double) {
      shifting_rep = multiply_power10_64bit(shifting_rep, exp10_mag);
    } else {
      shifting_rep = multiply_power10_32bit(shifting_rep, exp10_mag);
    }

    // Final bit shift: Shift may be large, guard against UB
    return guarded_right_shift(shifting_rep, exp2_mag);
  };

  // If our total decimal shift is less than the max, we don't need to iterate
  if (exp10_mag <= Constants::max_digits_shift) {
    return final_shifts_low10s();
  }

  // We want to start by lining up our bits to num_rep_bits, but since we'll be bit-shifting
  // down, we need even more low bits as a buffer (see comments in exp10 > 0 function)
  // This would normally just be normal_lineup_shift as in the other function, but 
  // that assumed that the understood floating-point bit was set, and this does not. 
  // Why not? Because our input could be a denormal! Which it can't be for exp10 > 0. 
//  auto const lineup_shift       = Constants::num_rep_bits - count_significant_bits(base2_value);
//  int const num_init_bit_shift = lineup_shift + Constants::num_2s_shift_buffer_bits;

int const shift_up_to = sizeof(ShiftingRep) * 8 - Constants::max_bits_shift; //4 ok, 1 bad
int const shift_from = count_significant_bits(base2_value);
int const num_init_bit_shift = shift_up_to - shift_from;

  // Constants::num_2s_shift_buffer_bits; Note: This shift is safe to do in the smaller IntegerRep
  // as it is up to bit 61 / 31
  shifting_rep <<= num_init_bit_shift;
  exp2_mag += num_init_bit_shift;

  // Iterate, multiplying by 10s and shifting down by 2s until we're almost done
  do {
    // More decimal places to shift than we have room: Multiply the max number of 10s
    shifting_rep *= Constants::max_digits_shift_pow;
    exp10_mag -= Constants::max_digits_shift;

    // If our remaining bit shift is less than the max, we're finished iterating
    if (exp2_mag <= Constants::max_bits_shift) {
      // Last bit-shift: Shift all remaining bits, apply the remaining scale, then bail
      shifting_rep >>= exp2_mag;

      // We need to convert to the output rep for the final scale-factor multiply, because if (e.g.)
      // float -> dec128 and some large exp10_mag, it might overflow the 64bit shifting rep.
      // It's not needed for exp10 > 0 because we're dividing by 10s there instead of multiplying.
      using UnsignedRep = cuda::std::make_unsigned_t<Rep>;
      // NOTE: This can overflow! (Both multiply and cast)
      return multiply_power10<UnsignedRep>(static_cast<UnsignedRep>(shifting_rep), exp10_mag);
    }

    // More bits to shift than we have room: Shift the max number of 2s
    shifting_rep >>= Constants::max_bits_shift;
    exp2_mag -= Constants::max_bits_shift;
  } while (exp10_mag > Constants::max_digits_shift);

  // Do our final shifts
  return final_shifts_low10s();
}

/**
 * @brief Perform lossless floating-point -> integer decimal conversion
 *
 * @note Info is lost if the chosen scale factor truncates information.
 *
 * @tparam Rep The type of integer we are converting to, to store the decimal value
 * @tparam FloatingType The type of floating-point object we are converting from
 * @param floating The floating point value to convert
 * @param scale The desired base-10 scale factor: decimal value = returned value * 10^scale
 * @return Integer representation of the floating-point value, given the desired scale
 */
template <typename Rep,
          typename FloatingType,
          typename cuda::std::enable_if_t<cuda::std::is_floating_point_v<FloatingType>>* = nullptr>
CUDF_HOST_DEVICE inline Rep convert_floating_to_integral(FloatingType const& floating,
                                                         scale_type const& scale)
{
  // Extract components of the floating point number
  using converter        = floating_converter<FloatingType>;
  auto const integer_rep = converter::bit_cast_to_integer(floating);
  if (converter::is_zero(integer_rep)) { return 0; }

  // Note that the base2_value here is an unsigned integer with sizeof(FloatingType)
  auto const is_negative                  = converter::get_is_negative(integer_rep);
  auto const [significand, floating_exp2] = converter::get_significand_and_exp2(integer_rep);

  // Add half of a floating-point bit to our value to account for rounding. 
  // See function comments for discussion. 
  auto const exp10 = static_cast<int>(scale);
  auto const [base2_value_bound, exp2_bound] = add_half_bit(significand, floating_exp2, exp10);

  //Structured bindings cannot be captured :/
  auto const base2_value = base2_value_bound;
  auto const exp2 = exp2_bound;

  // Apply the powers of 2 and 10 to convert to decimal.
  // The result will be incremented * (2^exp2) / (10^exp10)
  //
  // Note that while this code is branchy, the decimal scale factor is part of the
  // column type itself, so every thread will take the same branches on exp10.
  // Also data within a column tends to be similar, so they will often take the
  // same branches on exp2 as well.
  //
  // NOTE: some returns here can overflow (e.g. ShiftingRep -> UnsignedRep)
  using UnsignedRep = cuda::std::make_unsigned_t<Rep>;
  auto const magnitude = [&]() -> UnsignedRep {

    if (exp10 == 0) {
      // NOTE: Left Bit-shift can overflow! As can cast! (e.g. double -> decimal32)
      // Bit shifts may be large, guard against UB
      if (exp2 >= 0) {
        return guarded_left_shift(static_cast<UnsignedRep>(base2_value), exp2);
      } else {
        return guarded_right_shift(base2_value, -exp2);
      }
    } else if (exp10 > 0) {
      if (exp2 <= 0) {
        // Power-2/10 shifts both downward: order doesn't matter, apply and bail.
        // Guard against shift being undefined behavior
        auto const shifted = guarded_right_shift(base2_value, -exp2);
        return divide_power10<decltype(shifted)>(shifted, exp10);
      }
      return shift_to_decimal_posexp<FloatingType>(base2_value, exp2, exp10);
    } else {  // exp10 < 0
      if (exp2 >= 0) {
        // Power-2/10 shifts both upward: order doesn't matter, apply and bail.
        // NOTE: Either shift, multiply, or cast (e.g. double -> decimal32) can overflow!
        auto const shifted = guarded_left_shift(static_cast<UnsignedRep>(base2_value), exp2);
        return multiply_power10<UnsignedRep>(shifted, -exp10);
      }
      return shift_to_decimal_negexp<Rep, FloatingType>(base2_value, exp2, exp10);
    }
  }();

  // Floor the magnitude to the appropriate decimal place. 
  // This truncates leftover data from add_half_bit(). See that function for details. 
  auto const floored = [&](){
  // So when does the user-supplied scale truncate info?
  // If exp2 > 0 (left-shift), truncate if divide by more 10s (exp10 > 0) than we shift up by 2s
  // If exp2 < 0 (right-shift), truncate if multiply by LESS OR SAME 10s (exp10 < 0) than we shift down by 2s
//because 2^10 > 10^3 (& 2^299 > 10^90), corresponding_exp10 will always be slightly small
//so if exp2 = 10 & exp10 = 3, we will still want to add one extra digit
/*
    int const corresponding_exp10 = 90 * exp2 / 299; //2^-50 -> -15.05
    //want to round down for -powers of 10, but round UP for positive!
    int const floor_power = corresponding_exp10 - exp10; //-15 + 18 = 3

    int const floor_power = 90 * exp2 / 299 - exp10; //always round this up
//how to round up? add +0.5 and floor
    int const floor_power = (90 * exp2 - 299 * exp10) / 299; //always round this up
//how to add 0.5? add 
    int const floor_power = (2 * (90 * exp2 - 299 * exp10) + 299) / 598; //always round this down
*/
    int const floor_power = (90 * exp2 - 299 * exp10 + 299) / 299; //always round this down
    if(floor_power <= 0) {
      return magnitude;
    }

    auto const truncated = divide_power10<Rep>(magnitude, floor_power);
    return multiply_power10<Rep>(truncated, floor_power);
  }();

  // Reapply the sign and return
  // NOTE: Cast can overflow!
  auto const signed_magnitude = static_cast<Rep>(floored);
  return is_negative ? -signed_magnitude : signed_magnitude;
}

/**
 * @brief Perform (nearly) lossless base-10 -> base-2 fixed-point conversion for exp10 > 0
 *
 * @note Intended to only be called internally.
 *
 * @tparam DecimalRep The decimal integer type we are converting from
 * @tparam FloatingType The type of floating point object we are converting to
 * @param decimal_rep The decimal integer to convert
 * @param exp10 The number of powers of 10 to apply to undo the scale factor.
 * @return A pair of the base-2 value and the remaining powers of 2 to be applied.
 */
template <typename FloatingType,
          typename DecimalRep,
          typename cuda::std::enable_if_t<cuda::std::is_floating_point_v<FloatingType>>* = nullptr>
CUDF_HOST_DEVICE inline auto shift_to_binary_posexp(DecimalRep decimal_rep, int exp10)
{
  // This is the reverse of shift_to_decimal_posexp(), see that for more details.

  // ShiftingRep: uint64 for float's, __uint128_t for double's
  using Constants   = shifting_constants<FloatingType>;
  using ShiftingRep = typename Constants::ShiftingRep;

  // We would start by lining up our data to num_rep_bits, but since we'll be bit-shifting
  // down, we need even more low bits as a buffer (see comments on these constants)
//  auto const num_significant_bits = count_significant_bits(decimal_rep);
//  int exp2 = num_significant_bits - (Constants::num_rep_bits + Constants::num_2s_shift_buffer_bits);

int const shift_up_to = sizeof(ShiftingRep) * 8 - Constants::max_bits_shift;
int const shift_from = count_significant_bits(decimal_rep);
int const num_init_bit_shift = shift_up_to - shift_from;
int exp2 = -num_init_bit_shift;

  // Perform the initial bit shift
  ShiftingRep shifting_rep;
  if constexpr (sizeof(ShiftingRep) < sizeof(DecimalRep)) {
    // Shift within DecimalRep before dropping to the smaller ShiftingRep
    decimal_rep  = (exp2 >= 0) ? (decimal_rep >> exp2) : (decimal_rep << -exp2);
    shifting_rep = static_cast<ShiftingRep>(decimal_rep);
  } else {
    // Scale up to ShiftingRep before shifting
    shifting_rep = static_cast<ShiftingRep>(decimal_rep);
    shifting_rep = (exp2 >= 0) ? (shifting_rep >> exp2) : (shifting_rep << -exp2);
  }

  // Iterate, multiplying by 10s and shifting down by 2s until we're almost done
  while (exp10 > Constants::max_digits_shift) {
    // More decimal places to shift than we have room: Multiply the max number of 10s
    shifting_rep *= Constants::max_digits_shift_pow;
    exp10 -= Constants::max_digits_shift;

    // Then make more room by bit shifting down by the max # of 2s
    shifting_rep >>= Constants::max_bits_shift;
    exp2 += Constants::max_bits_shift;
  }

  // Last 10s-shift: multiply all remaining decimal places
  // The multiplier is less than the max-shift, and thus fits within 64 / 32 bits
  if constexpr (Constants::is_double) {
    shifting_rep = multiply_power10_64bit(shifting_rep, exp10);
  } else {
    shifting_rep = multiply_power10_32bit(shifting_rep, exp10);
  }

  // Our shifting_rep is now the integer mantissa, return it and the powers of 2
  return std::pair{shifting_rep, exp2};
}

/**
 * @brief Perform (nearly) lossless base-10 -> base-2 fixed-point conversion for exp10 < 0
 *
 * @note Intended to only be called internally.
 * @note A 1-ulp loss may occur, but only for magnitudes E-270 or smaller.
 *
 * @tparam DecimalRep The decimal integer type we are converting from
 * @tparam FloatingType The type of floating point object we are converting to
 * @param decimal_rep The decimal integer to convert
 * @param exp10 The number of powers of 10 to apply to undo the scale factor.
 * @return A pair of the base-2 value and the remaining powers of 2 to be applied.
 */
template <typename FloatingType,
          typename DecimalRep,
          typename cuda::std::enable_if_t<cuda::std::is_floating_point_v<FloatingType>>* = nullptr>
CUDF_HOST_DEVICE inline auto shift_to_binary_negexp(DecimalRep decimal_rep, int const exp10)
{
  // This is the reverse of shift_to_decimal_negexp(), see that for more details.

  // ShiftingRep: uint64 for float's, __uint128_t for double's
  using Constants   = shifting_constants<FloatingType>;
  using ShiftingRep = typename Constants::ShiftingRep;

  // We would start by lining up our data to num_rep_bits, but if we originated with a floating
  // number, we had to keep track of extra bits on the low-side because we were bit-shifting down.
  //
  // Those bits were not rounded. If we didn't truncate those bits before, we don't want to now
  // either to ensure that we end up at the same floating point value that we started with.
  //
  // We could try to round here instead, but we don't know if we came from a floating-point value
  // or not, so any rounding may not be desired.
  //
  // Note that here we are bit-shifting up, so we also need num_2s_shift_buffer_bits
  // room on the high side. We have barely enough room for this for floats, but we're one bit
  // over for doubles. So for doubles we'll keep one less bit on the low-side.
  //
  // This MAY cause a discrepancy in the last bit of our double from the value that we started with.
  // However we only need all 4 bits for extremely large exponents
  // (one bit to start + one extra bit every 90 powers of 10, so < E-270).
  // And it's only a partial bit, and the eventual cast to double rounds, so we
  // are often (always?) fine anyway (e.g. DBL_MIN & DBL_TRUE_MIN work fine).
  //
  // See comments on these constants for more details.
//  auto const num_significant_bits = count_significant_bits(decimal_rep);
//  int exp2 = num_significant_bits - (Constants::num_rep_bits);// + Constants::num_2s_shift_buffer_bits);
//  if constexpr (Constants::is_double) { ++exp2; }

  // Max bit shift left to give us the most room for shifting 10s: Multiply by 2s
//  exp2 -= Constants::max_bits_shift;


int const shift_up_to = sizeof(ShiftingRep) * 8 - Constants::num_2s_shift_buffer_bits;
int const shift_from = count_significant_bits(decimal_rep);
int const num_init_bit_shift = shift_up_to - shift_from;
int exp2 = -num_init_bit_shift;



  // Perform the initial bit shift
  ShiftingRep shifting_rep;
  if constexpr (sizeof(ShiftingRep) < sizeof(DecimalRep)) {
    // Shift within DecimalRep before dropping to the smaller ShiftingRep
    decimal_rep  = (exp2 >= 0) ? (decimal_rep >> exp2) : (decimal_rep << -exp2);
    shifting_rep = static_cast<ShiftingRep>(decimal_rep);
  } else {
    // Scale up to ShiftingRep before shifting
    shifting_rep = static_cast<ShiftingRep>(decimal_rep);
    shifting_rep = (exp2 >= 0) ? (shifting_rep >> exp2) : (shifting_rep << -exp2);
  }

  // Convert to using positive values upfront, simpler than doing later.
  int exp10_mag = -exp10;

  // Iterate, dividing by 10s and shifting up by 2s until we're almost done
  while (exp10_mag > Constants::max_digits_shift) {
    // More decimal places to shift than we have room: Divide the max number of 10s
    // Note that the result of this division is guaranteed to fit within low 64/32 bits
    // See discussion in shift_to_decimal_posexp() for more details
    shifting_rep /= Constants::max_digits_shift_pow;
    exp10_mag -= Constants::max_digits_shift;

    // Then make more room by bit shifting up by the max # of 2s
    shifting_rep <<= Constants::max_bits_shift;
    exp2 -= Constants::max_bits_shift;
  }

  // Last 10s-shift: Divdie all remaining decimal places.
  // This divide result may not fit in the low half of the bit range
  // But the divisor is less than the max-shift, and thus fits within 64 / 32 bits
  if constexpr (Constants::is_double) {
    shifting_rep = divide_power10_64bit(shifting_rep, exp10_mag);
  } else {
    shifting_rep = divide_power10_32bit(shifting_rep, exp10_mag);
  }

  // Our shifting_rep is now the integer mantissa, return it and the powers of 2
  return std::pair{shifting_rep, exp2};
}

/**
 * @brief Perform (nearly) lossless integer decimal -> floating-point conversion
 *
 * @note A 1 ulp loss may occur, but only to doubles with magnitude <= 1E-270
 *
 * @tparam FloatingType The type of floating-point object we are converting to
 * @tparam Rep The decimal integer type we are converting from
 * @param value The decimal integer to convert
 * @param scale The base-10 scale factor for the input integer
 * @return Floating-point representation of the scaled integral value
 */
template <typename FloatingType,
          typename Rep,
          typename cuda::std::enable_if_t<cuda::std::is_floating_point_v<FloatingType>>* = nullptr>
CUDF_HOST_DEVICE inline FloatingType convert_integral_to_floating(Rep const& value,
                                                                  scale_type const& scale)
{
  // Check the sign of the input
  bool const is_negative = (value < 0);

  // Convert to unsigned for bit counting/shifting
  using UnsignedType        = cuda::std::make_unsigned_t<Rep>;
  auto const unsigned_value = [&]() -> UnsignedType {
    // Use built-in abs functions where available
    if constexpr (cuda::std::is_same_v<Rep, int64_t>) {
      return cuda::std::llabs(value);
    } else if constexpr (!cuda::std::is_same_v<Rep, __int128_t>) {
      return cuda::std::abs(value);
    }

    // No abs function for 128bit types, so have to do it manually.
    // Must guard against minimum value, as we can't just negate it: not representable.
    if (value == cuda::std::numeric_limits<__int128_t>::min()) {
      return static_cast<UnsignedType>(value);
    } else {
      return static_cast<UnsignedType>(is_negative ? -value : value);
    }
  }();

  // Shift by powers of 2 and 10 to get our integer mantissa
  auto const [mantissa, exp2] = [&]() {
    auto const exp10 = static_cast<int32_t>(scale);
    if (exp10 >= 0) {
      return shift_to_binary_posexp<FloatingType>(unsigned_value, exp10);
    } else {  // exp10 < 0
      return shift_to_binary_negexp<FloatingType>(unsigned_value, exp10);
    }
  }();

  // Zero has special exponent bits, just handle it here
  if (mantissa == 0) { return FloatingType(0.0f); }

  // Cast our integer mantissa to floating point
  auto const floating = static_cast<FloatingType>(mantissa);  // IEEE-754 rounds to even

  // Apply the sign and the remaining powers of 2
  using converter      = floating_converter<FloatingType>;
  auto const magnitude = converter::add_exp2(floating, exp2);
  return converter::set_is_negative(magnitude, is_negative);
}

}  // namespace detail

/** @} */  // end of group
}  // namespace numeric
