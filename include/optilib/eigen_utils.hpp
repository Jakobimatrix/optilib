#ifndef EIGEN_UTILS_HPP
#define EIGEN_UTILS_HPP

#include <Eigen/Dense>
#include <array>

#include "constraints.hpp"
#include "definitions.hpp"
#include "rand.hpp"

namespace opt {

template <unsigned n, class T = double>
using XdPoint = Eigen::Matrix<T, n, 1>;


template <unsigned n, class T = double>
T squaredNorm(const XdPoint<n, T>& p) {
  T normsquared = 0;
  for (unsigned i = 0; i < n; ++i) {
    normsquared += p(i, 0) * p(i, 0);
  }
  return normsquared;
}

// Struct to store information about a n-D line between two points p1 and p2.
template <unsigned n, class T>
struct HyperLine {
  using Point = XdPoint<n, T>;

  // line = (p2 - p1)*s + p1
  Point p1;
  Point p2;

  /*!
   * \brief Calculates a point on (p2 - p1)*s + p1 given s.
   * \param s The distance variable from the line.
   * \return The requested point on the line.
   */
  Eigen::VectorXd getPoint(const double s) const noexcept {
    return (p2 - p1) * s + p1;
  }

  /*!
   * \brief Given a point calculate the closest point on this line.
   * \param p The point from which we want to know the closest point on this
   * Line. \return The Point on this Line closest to p.
   */
  Point getClosestPointOnLine(const Point& p) const noexcept {
    /*
    clang-format off
    Closest point pc to p on l(x) = (p2 - p1)*x + p1
    And v1 = (p2 - p1)
    And v2 = (pL - p)
    Than v1 * v2 = 0 (orthogonal)
    pc is somewere on l1 -> pc = l(x) = (p2 - p1)*x + p1

    0 = Summ[ v1_i * v2_i]
      = Summ[ v1_i * (pc - p)_i]
      = Summ[ v1_i * (l(x) - p)_i]
      = Summ[ v1_i * ((p2 - p1)*x + p1 - p)_i]
      = Summ[ v1_i * (v1*x + p1 - p)_i]
      = Summ[ x * v1_i^2 + v1_i^2 * (p1 - p)_i]
    Summ[ x * v1_i^2 ] = Summ[ v1_i^2 * (p1 - p)_i ]
    x * norm(v1)^2 = transposed(v1)*(p1 - p)
    x = transposed(v1)*(p1 - p)/norm(v1)^2
    clang-format on
    */

    const auto v1 = p2 - p1;
    const auto x = v1.dot(p1 - p) / squaredNorm<n, T>(v1);
    return getPoint(-x);  // TODO where does the -1 come from?
  }
};

template <unsigned p, unsigned q, class T>
Eigen::Matrix<T, p, q> getUniformRandMatrix(T min, T max) {
  const auto rg = UniformDistribution(min, max);
  return Eigen::Matrix<T, p, q>::Zero().unaryExpr(
      [&](float dummy) { return rg->get(); });
}

template <unsigned p, class T>
ObjectiveType<p, T> getRandObjective(
    T min, T max, const std::vector<std::shared_ptr<Constraint<p, true, T>>>& constraints) {
  const auto rg = UniformDistribution(min, max);
  if (constraints.empty()) {
    return ObjectiveType<p, T>::Zero().unaryExpr(
        [&](float dummy) { return rg->get(); });
  }
}

template <unsigned p, class T>
ObjectiveType<p, T> getRandObjective(
    const ObjectiveType<p, T>& min,
    const ObjectiveType<p, T>& max,
    const std::vector<std::shared_ptr<Constraint<p, true, T>>>& constraints) {
  auto rg = RandomGenerator::getInstance();
  ObjectiveType<p, T> o;
  for (unsigned i = 0; i < p; ++i) {
    o(i, 0) = rg->uniformDistribution(min(i, 0), max(1.0));
  }
  if (constraints.empty()) {
    return o;
  }
}



}  // namespace opt

#endif  // EIGEN_UTILS_HPP
