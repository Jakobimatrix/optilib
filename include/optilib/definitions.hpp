#ifndef OPTIMIZATION_DEFINITIONS
#define OPTIMIZATION_DEFINITIONS
#include <Eigen/Dense>
#include <functional>

namespace opt {
// Objective O in R^p
template <unsigned p, typename T = double>
using ObjectiveType = Eigen::Matrix<T, p, 1>;
// in R^dim

// Objective Function J(x_1,x_2,..,x_{p}) in R^p -> R
template <unsigned p, typename T = double>
using ObjectiveFunctionType = std::function<T(const ObjectiveType<p> &)>;

// dJ/dO in R^p -> R^p
template <unsigned p, typename T = double>
using ObjectiveFunctionDerivativeType =
    std::function<ObjectiveType<p, T>(const ObjectiveType<p, T> &)>;

// <EnableType>
template <bool has_derivative, typename Enable = void>
struct EnableType {};

template <bool has_derivative>
struct EnableType<has_derivative, std::enable_if<has_derivative>> {
  typedef bool type;
};

template <bool has_derivative>
struct EnableType<has_derivative, std::enable_if<!has_derivative>> {};
// </EnableType>

using TrueType = EnableType<true>;
using FalseType = EnableType<false>;

}  // namespace opt
#endif
