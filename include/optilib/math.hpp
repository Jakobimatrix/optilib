#ifndef MATH_HPP
#define MATH_HPP

#include <cmath>

namespace opt {
template <class T>
inline bool isNearlyEqual(T a, T b) {
  constexpr int factor = 7;

  T min_a = a - (a - std::nextafter(a, std::numeric_limits<T>::lowest())) * factor;
  T max_a = a + (std::nextafter(a, std::numeric_limits<T>::max()) - a) * factor;

  return min_a <= b && max_a >= b;
}

template <class T>
inline bool isNearlyZero(T v) {
  constexpr int factor = 7;
  constexpr T zero(static_cast<T>(0.));

  /*constexpr*/ const T min =
      -std::nextafter(zero, std::numeric_limits<T>::lowest()) * factor;
  /*constexpr*/ const T max = std::nextafter(zero, std::numeric_limits<T>::max()) * factor;

  return min <= v && max >= v;
}

template <class T>
constexpr bool isNearlyZero(T value, T epsilon) {
  return -epsilon <= value && value <= epsilon;
}


// https://stackoverflow.com/questions/17719674/c11-fast-constexpr-integer-powers
constexpr int64_t ipow_(int base, int exp) {
  return exp > 1 ? ipow_(base, (exp >> 1) + (exp & 1)) * ipow_(base, exp >> 1) : base;
}
constexpr int64_t ipow(int base, int exp) {
  return exp < 1 ? 1 : ipow_(base, exp);
}

// Get the smallest epsilon you can use arithmetic with the given val with, without beeing treated as zero.
template <class T>
constexpr T getEpsilon(T val) {
  const double epsilon_m = std::nextafter(val, std::numeric_limits<double>::lowest());
  const double epsilon_M = std::nextafter(val, std::numeric_limits<double>::max());
  return std::max(epsilon_M - val, val - epsilon_m);
}

}  // namespace opt

#endif  // MATH_HPP
