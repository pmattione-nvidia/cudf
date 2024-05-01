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


#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/utilities/traits.hpp>

#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <cstring> //memcpy

namespace numeric {

/**
 * @addtogroup floating_conversion
 * @{
 * @file
 * @brief fixed_point <--> floating-point conversion functions. 
 */

namespace detail {

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

CUDF_HOST_DEVICE inline uint64_t decimal_shift_right(__uint128_t num)
{
  // Modified from libdivide_128_div_64_to_64() in libdivide, because here
  // we know the denominator at compile time (10^18). 
  // From https://github.com/ridiculousfish/libdivide/blob/master/libdivide.h
  // Used under the zlib license:

  // zlib License
  // ------------

  // Copyright (C) 2010 - 2019 ridiculous_fish, <libdivide@ridiculousfish.com>
  // Copyright (C) 2016 - 2019 Kim Walisch, <kim.walisch@gmail.com>

  // This software is provided 'as-is', without any express or implied
  // warranty.  In no event will the authors be held liable for any damages
  // arising from the use of this software.

  // Permission is granted to anyone to use this software for any purpose,
  // including commercial applications, and to alter it and redistribute it
  // freely, subject to the following restrictions:

  // 1. The origin of this software must not be misrepresented; you must not
  //    claim that you wrote the original software. If you use this software
  //    in a product, an acknowledgment in the product documentation would be
  //    appreciated but is not required.
  // 2. Altered source versions must be plainly marked as such, and must not be
  //    misrepresented as being the original software.
  // 3. This notice may not be removed or altered from any source distribution.

  uint64_t numhi = num >> 64;
  uint64_t numlo = static_cast<uint64_t>(num);

  // We work in base 2**32.
  // A uint32 holds a single digit. A uint64 holds two digits.
  // Our numerator is conceptually [num3, num2, num1, num0].
  // Our denominator is [den1, den0].

  // The high and low digits of our computed quotient.
  uint32_t q1;
  uint32_t q0;

  // The normalization shift factor.
  constexpr int shift = 4; //10^18 in binary has 4 leading zeroes

  // The high and low digits of our numerator (after normalizing).
  uint32_t num1;
  uint32_t num0;

  // A partial remainder.
  uint64_t rem;

  // The estimated quotient, and its corresponding remainder (unrelated to true remainder).
  uint64_t qhat;
  uint64_t rhat;

  // Variables used to correct the estimated quotient.
  uint64_t c1;
  uint64_t c2;

  // Determine the normalization factor. We multiply den by this, so that its leading digit is at
  // least half b. In binary this means just shifting left by the number of leading zeros, so that
  // there's a 1 in the MSB.
  // We also shift numer by the same amount. This cannot overflow because numhi < den.
  // The expression (-shift & 63) is the same as (64 - shift), except it avoids the UB of shifting
  // by 64. The funny bitwise 'and' ensures that numlo does not get shifted into numhi if shift is
  // 0. clang 11 has an x86 codegen bug here: see LLVM bug 50118. The sequence below avoids it.
  static constexpr uint64_t original_den = 1000000000000000000ULL; //10^18
  static constexpr uint64_t den = original_den << shift;
  numhi <<= shift;
  numhi |= (numlo >> 60);
  numlo <<= shift;

  // Extract the low digits of the numerator and both digits of the denominator.
  num1 = (uint32_t)(numlo >> 32);
  num0 = (uint32_t)(numlo & 0xFFFFFFFFu);
  static constexpr auto den1 = (uint32_t)(den >> 32);
  static constexpr auto den0 = (uint32_t)(den & 0xFFFFFFFFu);

  // We wish to compute q1 = [n3 n2 n1] / [d1 d0].
  // Estimate q1 as [n3 n2] / [d1], and then correct it.
  // Note while qhat may be 2 digits, q1 is always 1 digit.
  qhat = numhi / den1;
  rhat = numhi % den1;
  c1 = qhat * den0;
  c2 = (rhat << 32) + num1;
  if (c1 > c2) qhat -= (c1 - c2 > den) ? 2 : 1;
  q1 = (uint32_t)qhat;

  // Compute the true (partial) remainder.
  rem = (numhi << 32) + num1 - q1 * den;

  // We wish to compute q0 = [rem1 rem0 n0] / [d1 d0].
  // Estimate q0 as [rem1 rem0] / [d1] and correct it.
  qhat = rem / den1;
  rhat = rem % den1;
  c1 = qhat * den0;
  c2 = (rhat << 32) + num0;
  if (c1 > c2) qhat -= (c1 - c2 > den) ? 2 : 1;
  q0 = (uint32_t)qhat;

  return ((uint64_t)q1 << 32) | q0;
}

/**
 * @brief Helper struct for getting and setting the components of a floating-point value
 *
 * @tparam FloatingType Type of floating-point value
 */
template <typename FloatingType,
          typename cuda::std::enable_if_t<cuda::std::is_floating_point_v<FloatingType>>* = nullptr>
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
  // The mantissa is normalized. There is an implicit 1 bit to the left of the binary point.
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
  // the power-of-2 is exponent + bias. The bias is 126 for floats and 1022 for doubles.
  /// 126 / 1022 for float / double
  static constexpr IntegralType exponent_bias =
    cuda::std::numeric_limits<FloatingType>::max_exponent - 2;

  /** @brief Reinterpret the bits of a floating-point value as an integer
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

  /** @brief Reinterpret the bits of an integer as floating-point value
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

  /** @brief Extracts the integral significand of a floating-point number
   *
   * @param floating The floating value to extract the significand from
   * @return The integral significand, bit-shifted to a (large) whole number
   */
  CUDF_HOST_DEVICE inline static IntegralType get_base2_value(FloatingType floating)
  {
    // Convert floating to integer
    auto const integer_rep = bit_cast_to_integer(floating);

    // Extract the significand, setting the high bit for the understood 1/2
    return (integer_rep & mantissa_mask) | understood_bit_mask;
  }

  /** @brief Extracts the sign bit of a floating-point number
   *
   * @param floating The floating value to extract the sign from
   * @return The sign bit
   */
  CUDF_HOST_DEVICE inline static bool get_is_negative(FloatingType floating)
  {
    // Convert floating to integer
    auto const integer_rep = bit_cast_to_integer(floating);

    // Extract the sign bit:
    return static_cast<bool>(sign_mask & integer_rep);
  }

  /** @brief Extracts the exponent of a floating-point number
   *
   * @note This returns INT_MIN for +/-0, +/-inf, NaN's, and denormals
   * For all of these cases, the decimal fixed_point number should be set to zero
   *
   * @param floating The floating value to extract the exponent from
   * @return The stored base-2 exponent, or INT_MIN for special values
   */
  CUDF_HOST_DEVICE inline static int get_exp2(FloatingType floating)
  {
    // Convert floating to integer
    auto const integer_rep = bit_cast_to_integer(floating);

    // First extract the exponent bits and handle its special values.
    // To minimize branching, all of these special cases will return INT_MIN.
    // For all of these cases, the decimal fixed_point number should be set to zero.
    auto const exponent_bits = integer_rep & exponent_mask;
    if (exponent_bits == 0) {
      // Because of the understood set-bit not stored in the mantissa, it is not possible
      // to store the value zero directly. Instead both +/-0 and denormals are represented with
      // the exponent bits set to zero.
      // Thus it's fastest to just floor (generally unwanted) denormals to zero.
      return INT_MIN;
    } else if (exponent_bits == exponent_mask) {
      //+/-inf and NaN values are stored with all of the exponent bits set.
      // As none of these are representable by integers, we'll return the same value for all cases.
      return INT_MIN;
    }

    // Extract the exponent value: shift the bits down and subtract the bias.
    using SignedIntegralType                       = cuda::std::make_signed_t<IntegralType>;
    SignedIntegralType const shifted_exponent_bits = exponent_bits >> num_mantissa_bits;
    return shifted_exponent_bits - static_cast<SignedIntegralType>(exponent_bias);
  }

  /** @brief Sets the sign bit of a floating-point number
   *
   * @param floating The floating value to set the sign of
   * @param is_negative The sign bit to set for the floating-point number
   * @return The input floating-point value with the chosen sign
   */
  CUDF_HOST_DEVICE inline static FloatingType set_is_negative(FloatingType floating,
                                                              bool is_negative)
  {
    // Convert floating to integer
    IntegralType integer_rep = bit_cast_to_integer(floating);

    // Set the sign bit
    integer_rep |= (IntegralType(is_negative) << sign_bit_index);

    // Convert back to float
    return bit_cast_to_floating(integer_rep);
  }

  /** @brief Adds to the base-2 exponent of a floating-point number
   *
   * @param floating The floating value to add to the exponent of
   * @param exp2 The power-of-2 to add to the floating-point number
   * @return The input floating-point value * 2^exp2
   */
  CUDF_HOST_DEVICE inline static FloatingType add_exp2(FloatingType floating, int exp2)
  {
    // Convert floating to integer
    auto integer_rep = bit_cast_to_integer(floating);

    // Extract the currently stored (biased) exponent
    auto exponent_bits = integer_rep & exponent_mask;
    auto stored_exp2   = exponent_bits >> num_mantissa_bits;

    // Add the additional power-of-2
    stored_exp2 += exp2;
    exponent_bits = stored_exp2 << num_mantissa_bits;

    // Clear existing exponent bits and set new ones
    integer_rep &= (~exponent_mask);
    integer_rep |= exponent_bits;

    // Convert back to float
    return bit_cast_to_floating(integer_rep);
  }
};

/** @brief Determine the number of significant bits in an integer
 *
 * @tparam T Type of input integer value
 * @param value The integer whose bits are being counted
 * @return The number of significant bits: the # of bits - # of leading zeroes
 */
template <typename T, typename cuda::std::enable_if_t<(cuda::std::is_unsigned_v<T>)>* = nullptr>
CUDF_HOST_DEVICE inline int count_significant_bits(T value)
{
#ifdef __CUDA_ARCH__
  if constexpr (std::is_same_v<T, uint64_t>) {
    return 64 - __clzll(static_cast<int64_t>(value));
  } else if constexpr (sizeof(T) <= sizeof(uint32_t)) {
    return 32 - __clz(static_cast<int32_t>(value));
  } else {
    // 128 bit type, must break u[ into high and low components
    auto const high_bits = static_cast<int64_t>(value >> 64);
    auto const low_bits  = static_cast<int64_t>(value);
    return 128 - (__clzll(high_bits) + int(high_bits == 0) * __clzll(low_bits));
  }
#else
  // Undefined behavior to call __builtin_clzll() with zero in gcc and clang
  if (value == 0) { return 0; }

  if constexpr (std::is_same_v<T, uint64_t>) {
    return 64 - __builtin_clzll(value);
  } else if constexpr (sizeof(T) <= sizeof(uint32_t)) {
    return 32 - __builtin_clz(value);
  } else {
    // 128 bit type, must break u[ into high and low components
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
 * @brief Helper struct with common constants needed by the floating <--> decimal conversions
 */
template <typename FloatingType>
struct shifting_constants
{
  // To convert a floating point number to a scaled decimal, we need to apply the powers of 2 
  static constexpr bool is_double = cuda::std::is_same_v<FloatingType, double>; ///< whether the type is double

  // Integer type that can hold the value of the significand
  using IntegerRep = std::conditional_t<is_double, uint64_t, uint32_t>;

  // Num bits needed to hold the significand
  static constexpr auto num_significand_bits = cuda::std::numeric_limits<FloatingType>::digits;

  // Shift data back and forth in space of a type with 2x the starting bits, to give us enough room
  using ShiftingRep = std::conditional_t<is_double, __uint128_t, uint64_t>;

  // The significand of a float / double is 24 / 53 bits
  // However, to uniquely represent each double / float as a different # in decimal format
  // you need 17 / 9 digits (from std::numeric_limits<T>::max_digits10)
  // To represent 10^17 / 10^9, you need 57 / 30 bits
  // So we need to keep track of this # of bits during shifting to ensure no info is lost
  static constexpr int num_rep_bits = is_double ? 57 : 30; ///< # bits to hold the value

  // We will be alternately shifting our data back and forth by powers of 2 and 10 to convert 
  // between floating and decimal (see shifting functions for details). 

  // For float -> decimal, we want to start with our significand bits at the top of the 
  // num_rep_bits range, so that we don't lose information we need on intermediary right-shifts
  static constexpr int lineup_shift = num_rep_bits - num_significand_bits;

  // However, to iteratively go back and forth, our 2s and 10s shifts must be of (nearly)
  // the same magnitude, or else our data will over-/under-flow. 

  // 2^10 (1024) is approximately 10^3, so the max shifts will be related by this 10/3 ratio
  // But, the difference between 2^10 and 10^3 is 1024/1000: 2.4%
  // So every time we shift by 10 bits and 3 decimal places, the 2s shift is an extra 2.4%

  // This 2.4% error compounds each time we do an iteration. 
  // The max float is about 2^128, approximately 3.4*10^38
  // With our 10/3 ratio, 128 * 3 / 10 = 38.4; 10^38.4 = 2.5*10^38, off by 1.36x
  // The max double is about 2^1024, which is approximately 1.8*10^308
  // With our 10/3 ratio, 1024 * 3 / 10 = 307.2: 10^307.2 = 1.6*10^307, off by 11.3x!

  // Rather than complicate our loop code to shift one less bit occassionaly (slowing us down), 
  // we'll instead use extra bits (in the direction we're shifting the 2s!) to compensate: 
  // 4 extra bits for doubles (2^4 = 16 > 11.3), 1 for floats (2 > 1.36)
  static constexpr int num_2s_shift_buffer_bits = is_double ? 4 : 1; ///< # bits for shifting error

  // How much room do we have for shifting? For double / float: 
  // 128 / 64 bits of space - 61 / 31 for rep + buffer = 67 / 33 bits for shifting
  // 2^67 / 2^33 are approximately 1.5*10^20 / 8.6*10^9
  // Thus we have room to shift up to 20 / 9 decimal places at once (for double/float)

  // However, we need to stick to our 10-bits / 3-decimals shift ratio to not over/under-flow
  // To simplify our loop code, we'll keep to this ratio exactly by rounding down to max
  // decimal shifts of 18 / 9 for double / float, and thus 60 / 30 bits
  static constexpr int max_digits_shift = is_double ? 18 : 9; ///< max decimal place shift
  static constexpr int max_bits_shift   = max_digits_shift * 10 / 3; ///< max bit shift: 60, 30

  // Pre-calculate 10^max_digits_shift. Note that it can fit within IntegerRep. 
  static constexpr auto max_digits_shift_pow = multiply_power10<IntegerRep>(IntegerRep(1), max_digits_shift); ///< 10^max-shift
};

/** @brief Increment integer rep of floating point if conversion causes truncation
 *
 * @note This fixes problems like 1.2 (value = 1.1999...) at scale -1 -> 11
 * 
 * @tparam T Type of integer holding the floating point significand.
 * @param significand The integer representation of the floating point significand
 * @param exp2 The power of 2 that needs to be applied to the significand. 
 * @param exp10 The power of 10 that needs to be applied to the significand. 
 * @return Incremented significand, but only if the conversion to integer causes truncation. 
 */
template <typename T,
          typename cuda::std::enable_if_t<cuda::std::is_integral_v<T>>* = nullptr>
CUDF_HOST_DEVICE T increment_on_truncation(T const significand, int const exp2, int const exp10)
{
  // The user-supplied scale may truncate information, so we need to talk about rounding. 
  // We have chosen not to round, so we want 1.23456f with scale -4 to be decimal 12345

  // But if we don't round at all, 1.2 (double) with scale -1 is 11 instead of 12! 
  // Why? Because 1.2 (double) is actually stored as 1.1999999... which we truncate to 1.1
  // While correct (given our choice to truncate), this is surprising and undesirable. 
  // This problem happens because 1.2 is not perfectly representable in floating point, 
  // and the value 1.199999... happened to be closer to 1.2 than the next value (1.2000...)

  // If the scale truncates information (we didn't choose to keep exactly 1.1999...), how 
  // do we make sure we store 1.2? All we have to do is add 1 ulp! (unit in the last place)
  // Then 1.1999... becomes 1.2000... which truncates to 1.2. 
  // And if it had been 1.2000..., adding 1 ulp still truncates to 1.2, the result is unchanged. 

  // The only way that this produces the incorrect result is if, when we entered 1.19999..., 
  // we truly meant 1.19999... (exactly, out to the very last bit), but then decided to truncate anyway. 
  // By choosing to truncate, you're saying you don't actually care about that level of precision, 
  // so being off by < 1 ulp should be just fine, compared to screwing up 1.2 with scale -1 -> 11 

  // So when does the user-supplied scale truncate info?
  // For powers > 0: When the 10s (scale) shift is larger than the corresponding bit-shift. 
  // For powers < 0: When the 10s shift is less than the corresponding bit-shift. 

  // Corresponding bit-shift: 
  // 2^10 is approximately 10^3, but this is off by 1.024%
  // 1.024^30 is 2.03704, so this is high by one bit for every 90 powers of 10
  // So for a power of 10 N (within double), 10^N = 2^(10*N/3 - N/90) = 2^(299*N/90)
  int const corresponding_exp2 = 299 * exp10 / 90;

  // If exp10 > 0, truncate if divide by more 10s than we shift up by 2s
  // If exp10 < 0, truncate if shift down by more OR THE SAME 2s than multiply by 10s
  // Truncate on the same: because for our approximation 2^299 > 10^90
  // Note that this works for both +/- exponents
  bool const conversion_truncates = (exp2 < corresponding_exp2) || 
    ((exp2 == corresponding_exp2) && (exp2 < 0));

  // (Potentially) apply increment and return
  return significand + conversion_truncates;
}

/** @brief Perform lossless base-2 -> base-10 fixed-point conversion for exp10 > 0
 *
 * @note Intended to only be called internally. 
 * @note The conversion is lossy if the chosen scale factor truncates information. 
 *
 * @tparam FloatingType The type of floating point object we are converting from
 * @param base2_value The base-2 fixed-point value we are converting from
 * @param exp2 The number of powers of 2 to apply to convert from base-2
 * @param exp10 The number of powers of 10 to apply to reach the desired scale factor
 * @return Magnitude of the converted-to decimal integer
 */
template <typename FloatingType,
          typename cuda::std::enable_if_t<cuda::std::is_floating_point_v<FloatingType>>* = nullptr>
CUDF_HOST_DEVICE inline auto shift_to_decimal_posexp10(
  typename shifting_constants<FloatingType>::IntegerRep const base2_value, int exp2, int exp10)
{
  // To convert to decimal, we need to apply the input powers of 2 and 10
  // The result will be (integer) base2_value * (2^exp2) / (10^exp10)
  // Output type is ShiftingRep

  // Adapted and heavily modified from Apache Arrow (Apache License 2.0): 
  // https://github.com/apache/arrow/blob/2babda0ba22740c092166b5c5d5d7aa9b4797953/cpp/src/arrow/util/decimal.cc#L90

  // Here exp10 > 0 and exp2 > 0, so we need to shift left by 2s and divide by 10s. 
  // To do this losslessly, we will iterate back and forth between them, shifting 
  // up by 2s and down by 10s until all of the powers have been applied. 

  // However the input base2_value type has virtually no spare room to shift our data
  // without over- or under-flowing and losing precision. 
  // So we'll cast up to ShiftingRep: uint64 for float's, __uint128_t for double's
  using Constants = shifting_constants<FloatingType>;
  using ShiftingRep = typename Constants::ShiftingRep;
  auto shifting_rep = static_cast<ShiftingRep>(base2_value);

  // We want to start by lining up our bits in num_rep_bits (see comments on lineup_shift), 
  // but since we start by bit-shifting anyway, try to combine the lineup_shift & max_bits_shift. 
  // We're shifting 2s up, so need num_2s_shift_buffer_bits space on the high side, which we do 
  // (our max bit shift is low enough that we don't shift into the highest bits)
  static constexpr int max_init_shift = Constants::lineup_shift + Constants::max_bits_shift;

  // If our total bit shift is less than this, we don't need to iterate
  if (exp2 <= max_init_shift) {
    // Shift bits left, divide by 10s to apply the scale factor, and we're done. 
    return divide_power10<ShiftingRep>(shifting_rep << exp2, exp10);
  }

  // We need to iterate. Do the combine lineup shift + iteration max shift
  shifting_rep <<= max_init_shift;
  exp2 -= max_init_shift;

  // Iterate, dividing by 10s and shifting up by 2s until we're almost done
  while(exp10 > Constants::max_digits_shift)
  {
    // More decimal places to shift than we have room: Divide the max number of 10s

    // Note that the result of this division is guaranteed to fit within low 64/32 bits
    // The highest set bit is num_rep_bits + num_2s_shift_buffer_bits + max_bits_shift
    // For float this is 30 + 1 + 30 = 61, for double 57 + 4 + 60 = 121
    // 2^61 / 10^9 (~2^30) is ~2^31, and 2^121 / 10^18 (~2^60) is ~2^61
    // Thus we can use a faster division routine that takes this account. 
    // I don't know of one for 64 by 32, but we can optimize for 128 by 64. 
    if constexpr (Constants::is_double)
      shifting_rep = decimal_shift_right(shifting_rep);
    else
      shifting_rep /= Constants::max_digits_shift_pow;
    exp10 -= Constants::max_digits_shift;

    // If our remaining bit shift is less than the max, we're finished iterating
    if (exp2 <= Constants::max_bits_shift) {
      // Shift bits left, divide by 10s to apply the scale factor, and we're done. 
      // This divide result may not fit in the low half of the bit range
      return divide_power10<ShiftingRep>(shifting_rep << exp2, exp10);
    }

    // Shift the max number of bits left again
    shifting_rep <<= Constants::max_bits_shift;
    exp2 -= Constants::max_bits_shift;
  }

  // Last 10s-shift: Divdie all remaining decimal places, shift all remaining bits, then bail
  // This divide result may not fit in the low half of the bit range
  // But the divisor is less than the max-shift, and thus fits within 64 / 32 bits
  if constexpr (Constants::is_double) {
    shifting_rep = divide_power10_64bit(shifting_rep, exp10);
  } else {
    shifting_rep = divide_power10_32bit(shifting_rep, exp10);
  }

  // Final bit shift
  return shifting_rep << exp2;
}

/** @brief Perform lossless base-2 -> base-10 fixed-point conversion for exp10 < 0
 *
 * @note Intended to only be called internally. 
 * @note The conversion is lossy if the chosen scale factor truncates information. 
 *
 * @tparam FloatingType The type of floating point object we are converting from
 * @param base2_value The base-2 fixed-point value we are converting from
 * @param exp2 The number of powers of 2 to apply to convert from base-2
 * @param exp10 The number of powers of 10 to apply to reach the desired scale factor
 * @return Magnitude of the converted-to decimal integer
 */
template <typename Rep, typename FloatingType, 
          typename cuda::std::enable_if_t<cuda::std::is_floating_point_v<FloatingType>>* = nullptr>
CUDF_HOST_DEVICE inline Rep shift_to_decimal_negexp10(
  typename shifting_constants<FloatingType>::IntegerRep const base2_value, int exp2, int exp10)
{
  // This is similar to shift_to_decimal_posexp10(), except exp10 < 0 & exp2 < 0
  // So instead here we need to multiply by 10s and shift right by 2s
  // See comments in that function for details. 

  // Convert to using positive values so we don't have keep negating each time we multiply
  int exp2_mag = -exp2;
  int exp10_mag = -exp10;

  // ShiftingRep: uint64 for float's, __uint128_t for double's
  using Constants = shifting_constants<FloatingType>;
  using ShiftingRep = typename Constants::ShiftingRep;
  ShiftingRep shifting_rep;

  // For performing final 10s-shift
  auto final_shifts_low10s = [&](){
    // Last 10s-shift: multiply all remaining decimal places, shift all remaining bits, then bail
    // The multiplier is less than the max-shift, and thus fits within 64 / 32 bits
    if constexpr (Constants::is_double) {
      shifting_rep = multiply_power10_64bit(shifting_rep, exp10_mag);
    } else {
      shifting_rep = multiply_power10_32bit(shifting_rep, exp10_mag);
    }
    
    // Final bit shift
    return shifting_rep >> exp2_mag;
  };

  // If our total decimal shift is less than the max, we don't need to iterate
  if (exp10_mag <= Constants::max_digits_shift) {
    shifting_rep = base2_value;
    return final_shifts_low10s();
  }

  // We want to start by lining up our bits to num_rep_bits, but since we'll be bit-shifting 
  // down, we need even more low bits as a buffer (see comments on these constants) 
  static constexpr int num_init_bit_shift = Constants::lineup_shift + Constants::num_2s_shift_buffer_bits;
  //Note: This shift is safe to do in the smaller IntegerRep as it is up to bit 61 / 31
  shifting_rep = base2_value << num_init_bit_shift;
  exp2_mag += num_init_bit_shift;

  // Iterate, multiplying by 10s and shifting down by 2s until we're almost done
  do
  {
    // More decimal places to shift than we have room: Multiply the max number of 10s
    shifting_rep *= Constants::max_digits_shift_pow;
    exp10_mag -= Constants::max_digits_shift;

    // If our remaining bit shift is less than the max, we're finished iterating
    if (exp2_mag <= Constants::max_bits_shift) {
      // Last bit-shift: Shift all remaining bits, apply the remaining scale, then bail
      shifting_rep >>= exp2_mag;

      // We need to convert to the output rep for the final scale-factor multiply, because if (e.g.) 
      // float -> dec128 and some tiny scale factor, it might overflow the 64bit shifting rep. 
      // It's not needed for exp10 > 0 because we're dividing by 10s there instead of multiplying. 
      using UnsignedRep = cuda::std::make_unsigned_t<Rep>;
      return multiply_power10<UnsignedRep>(UnsignedRep(shifting_rep), exp10_mag);
    }

    // More bits to shift than we have room: Shift the max number of 2s
    shifting_rep >>= Constants::max_bits_shift;
    exp2_mag -= Constants::max_bits_shift;
  } while (exp10_mag > Constants::max_digits_shift);
  
  // Do our final shifts
  return final_shifts_low10s();
}

/** @brief Perform floating-point -> integer decimal conversion (lossless for base-10). 
 *
 * @note Intended to only be called internally. 
 * @note The base-10 conversion is lossy if the scale factor truncates information. 
 *
 * @tparam Rep The type of integer we are converting to to store the decimal value. 
 * @tparam FloatingType The type of floating point object we are converting from. 
 * @param floating The floating point value to convert. 
 * @param scale The desired scale factor: decimal value = returned value * base^scale. 
 * @return Integer representation of the floating point value, given the desired scale.
 */
template <typename Rep,
          typename FloatingType,
          typename cuda::std::enable_if_t<cuda::std::is_floating_point_v<FloatingType>>* = nullptr>
CUDF_HOST_DEVICE inline Rep convert_floating_to_integral(FloatingType const& floating,
                                                         scale_type const& scale)
{
  // Extract components of the floating point number, see function for discussion
  // Note that the base2_value here is an (unsigned) integer
  using converter = floating_converter<FloatingType>;
  auto const is_negative = converter::get_is_negative(floating);
  auto const floating_exp2 = converter::get_exp2(floating);
  auto const base2_value = converter::get_base2_value(floating);

  // For special cases (+/-0, +/-inf, denormal, NaN): Return zero for all
  // Denormal: doing you a favor; +/-inf, NaN: not representable anyway. 
  if(floating_exp2 == INT_MIN) {
    return 0;
  }

  // Get the powers of 2 and 10 to apply to base2_value to convert it to decimal format
  // The result will be base2_value * (2^exp2) / (10^exp10)
  // To convert the significand to an integer, we effectively applied #-significand-bits 
  // powers of 2 to convert the fractional value to an integer, so subtract them off here
  using Constants = shifting_constants<FloatingType>;
  int const exp2 = floating_exp2 - Constants::num_significand_bits;
  auto const exp10 = static_cast<int>(scale);

  //Increment if truncating to yield expected value, see function for discussion
  auto const incremented = increment_on_truncation(base2_value, exp2, exp10);

  // Apply the powers of 2 and 10 to convert to decimal. 
  // Note that while this code is branchy, the decimal scale factor is part of the 
  // column type itself, so every thread will take the same branches on exp10. 
  // Also data within a column tends to be similar, so they will often take the 
  // same branches on exp2 as well. 
  auto const magnitude = [&]() -> Rep {
    if (exp10 == 0) { 
      return (exp2 >= 0) ? (Rep(incremented) << exp2) : Rep(incremented >> -exp2);
    } else if (exp10 > 0) {
      // If power-2/10 shifts both downward: order doesn't matter, apply and bail. 
      if(exp2 <= 0) { return divide_power10<decltype(incremented)>(incremented >> -exp2, exp10); }
      return shift_to_decimal_posexp10<FloatingType>(incremented, exp2, exp10);
    } else { //exp10 < 0
      // If power-2/10 shifts both upward: order doesn't matter, apply and bail. 
      using UnsignedRep = cuda::std::make_unsigned_t<Rep>;
      if (exp2 >= 0) { return multiply_power10<UnsignedRep>(UnsignedRep(incremented) << exp2, -exp10); }
      return shift_to_decimal_negexp10<Rep, FloatingType>(incremented, exp2, exp10);
    }
  }();

  // Reapply the sign and return
  return is_negative ? -magnitude : magnitude;
}

/** @brief Perform (nearly) lossless base-10 -> base-2 fixed-point conversion for exp10 > 0
 *
 * @note Intended to only be called internally. 
 *
 * @tparam DecimalRep The decimal integer type we are converting from
 * @tparam FloatingType The type of floating point object we are converting to
 * @param decimal_rep The decimal integer to convert
 * @param exp10 The number of powers of 10 to apply to undo the scale factor. 
 * @return A pair of the base-2 value and the remaining powers of 2 to be applied. 
 */
template <typename FloatingType, typename DecimalRep, 
          typename cuda::std::enable_if_t<cuda::std::is_floating_point_v<FloatingType>>* = nullptr>
CUDF_HOST_DEVICE inline auto shift_to_binary_posexp10(DecimalRep decimal_rep, int exp10)
{
  // This is the reverse of shift_to_decimal_posexp10(), see that for more details.

  // ShiftingRep: uint64 for float's, __uint128_t for double's
  using Constants = shifting_constants<FloatingType>;
  using ShiftingRep = typename Constants::ShiftingRep;

  // We would start by lining up our data to num_rep_bits, but since we'll be bit-shifting 
  // down, we need even more low bits as a buffer (see comments on these constants) 
  auto const num_significant_bits = count_significant_bits(decimal_rep);
  int exp2 = num_significant_bits - (Constants::num_rep_bits + Constants::num_2s_shift_buffer_bits);

  //Perform the initial bit shift
  ShiftingRep shifting_rep;
  if constexpr (sizeof(ShiftingRep) < sizeof(DecimalRep)) {
    //Shift within DecimalRep before dropping to the smaller ShiftingRep
    decimal_rep = (exp2 >= 0) ? (decimal_rep >> exp2) : (decimal_rep << -exp2);
    shifting_rep = static_cast<ShiftingRep>(decimal_rep);
  } else {
    //Scale up to ShiftingRep before shifting
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

/** @brief Perform (nearly) lossless base-10 -> base-2 fixed-point conversion for exp10 < 0
 *
 * @note Intended to only be called internally. 
 * @note A single-bit loss may occur, but only for magnitudes E-270 or smaller. 
 *
 * @tparam DecimalRep The decimal integer type we are converting from
 * @tparam FloatingType The type of floating point object we are converting to
 * @param decimal_rep The decimal integer to convert
 * @param exp10 The number of powers of 10 to apply to undo the scale factor. 
 * @return A pair of the base-2 value and the remaining powers of 2 to be applied. 
 */
template <typename FloatingType, typename DecimalRep, 
          typename cuda::std::enable_if_t<cuda::std::is_floating_point_v<FloatingType>>* = nullptr>
CUDF_HOST_DEVICE inline auto shift_to_binary_negexp10(DecimalRep decimal_rep, int const exp10)
{
  // This is the reverse of shift_to_decimal_negexp10(), see that for more details.

  // ShiftingRep: uint64 for float's, __uint128_t for double's
  using Constants = shifting_constants<FloatingType>;
  using ShiftingRep = typename Constants::ShiftingRep;

  // We would start by lining up our data to num_rep_bits, but if we originated with a floating
  // number, we had to keep track of extra bits on the low-side because we were bit-shifting down. 
  // We don't want to truncate those bits now to ensure that this ends up at the same floating
  // point value that we started with (assuming the scale factor didn't truncate anyway). 
  // See comments on these constants for more details. 
  auto const num_significant_bits = count_significant_bits(decimal_rep);
  int exp2 = num_significant_bits - (Constants::num_rep_bits + Constants::num_2s_shift_buffer_bits);
  if constexpr (Constants::is_double) {
    // Note that here we are bit-shifting up, so we also need num_2s_shift_buffer_bits
    // room on the high side. We have barely enough room for this for floats, but we're one bit
    // over for doubles. So for doubles we'll keep one less bit on the low-side. 
    
    // This MAY cause a discrepancy in the last bit of our double from the value
    // that we started with. However we only need all 4 bits for extremely large exponents
    // (one bit to start + one extra bit every 90 powers of 10, so < E-270). 
    // And it's only a partial bit, and the eventual cast to double rounds, so we 
    // are often fine anyway (e.g. DBL_MIN works fine). 
    ++exp2;
  }  
  // Max bit shift left to give us the most room for shifting 10s: Multiply by 2s
  exp2 -= Constants::max_bits_shift;

  //Perform the initial bit shift
  ShiftingRep shifting_rep;
  if constexpr (sizeof(ShiftingRep) < sizeof(DecimalRep)) {
    //Shift within DecimalRep before dropping to the smaller ShiftingRep
    decimal_rep = (exp2 >= 0) ? (decimal_rep >> exp2) : (decimal_rep << -exp2);
    shifting_rep = static_cast<ShiftingRep>(decimal_rep);
  } else {
    //Scale up to ShiftingRep before shifting
    shifting_rep = static_cast<ShiftingRep>(decimal_rep);
    shifting_rep = (exp2 >= 0) ? (shifting_rep >> exp2) : (shifting_rep << -exp2);
  }

  // Convert to using positive values upfront, simpler than doing later. 
  int exp10_mag = -exp10;

  // Iterate, dividing by 10s and shifting up by 2s until we're almost done
  while (exp10_mag > Constants::max_digits_shift) {

    // More decimal places to shift than we have room: Divide the max number of 10s
    // Note that the result of this division is guaranteed to fit within low 64/32 bits
    // See discussion in shift_to_decimal_posexp10() for more details
    if constexpr (Constants::is_double)
      shifting_rep = decimal_shift_right(shifting_rep);
    else
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

/** @brief Perform (nearly) lossless integer decimal -> floating-point conversion
 *
 * @note Intended to only be called internally. 
 * @note A single-bit loss may occur, but only for magnitudes E-270 or smaller. 
 *
 * @tparam FloatingType The type of floating point object we are converting to
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
  // If no scale to apply, cast and we're done
  auto const exp10 = static_cast<int32_t>(scale);
  if (exp10 == 0) { return static_cast<FloatingType>(value); }

  // Check the sign of the input
  bool const is_negative = (value < 0);

  // Convert to unsigned for bit counting/shifting
  using UnsignedType = cuda::std::make_unsigned_t<Rep>;
  auto const unsigned_value = [&]() -> UnsignedType {
    //Use built-in abs functions where available
    if constexpr (cuda::std::is_same_v<Rep, int64_t>) {
      return cuda::std::llabs(value);
    } else if constexpr (!cuda::std::is_same_v<Rep, __int128_t>) {
      return cuda::std::abs(value);
    }

    //No abs function for 128bit types, so have to do it manually. 
    //Must guard against minimum value, as we can't just negate it: not representable. 
    if(value == cuda::std::numeric_limits<__int128_t>::min()) {
      return static_cast<UnsignedType>(value);
    } else {
      return UnsignedType(is_negative ? -value : value);
    }
  }();
  
  // Shift by powers of 2 and 10 to get our integer significand
  auto const [significand, exp2] = [&](){
    if (exp10 > 0) {
      return shift_to_binary_posexp10<FloatingType>(unsigned_value, exp10);
    } else {  // exp10 < 0
      return shift_to_binary_negexp10<FloatingType>(unsigned_value, exp10);
    }
  }();

  //Zero has special exponent bits, just handle it here
  if(significand == 0) {
    return FloatingType(0.0f);
  }

  //Cast our integer significand to floating point
  auto const floating = static_cast<FloatingType>(significand);  // IEEE-754 rounds to even

  //Apply the sign and the remaining powers of 2
  using converter = floating_converter<FloatingType>;
  auto const magnitude = converter::add_exp2(floating, exp2);
  return converter::set_is_negative(magnitude, is_negative);
}

}  // namespace detail
}  // namespace numeric

