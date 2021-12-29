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

  constexpr T min = -std::nextafter(zero, std::numeric_limits<T>::lowest()) * factor;
  constexpr T max = std::nextafter(zero, std::numeric_limits<T>::max()) * factor;

  return min <= v && max >= v;
}

// https://stackoverflow.com/questions/17719674/c11-fast-constexpr-integer-powers
constexpr int64_t ipow_(int base, int exp) {
  return exp > 1 ? ipow_(base, (exp >> 1) + (exp & 1)) * ipow_(base, exp >> 1) : base;
}
constexpr int64_t ipow(int base, int exp) {
  return exp < 1 ? 1 : ipow_(base, exp);
}

}  // namespace opt

#endif  // MATH_HPP