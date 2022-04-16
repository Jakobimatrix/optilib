#define BOOST_TEST_MODULE stopping_test TestSuites
#define BOOST_TEST_DYN_LINK
#include <stdio.h>

#include <array>
#include <boost/test/unit_test.hpp>
#include <cmath>
#include <limits>
#include <numeric>
#include <optilib/eigen_utils.hpp>

namespace utf = boost::unit_test;
namespace tt = boost::test_tools;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdisabled-macro-expansion"

BOOST_AUTO_TEST_CASE(test_Hyperplane_form_conversation_3d, *utf::tolerance(0.00001)) {
  constexpr unsigned N = 3;
  using HyperPlane3d = opt::HyperPlane<N, double>;

  std::array<HyperPlane3d::Point, N> points;
  points[0] << 0, 0, 9;
  points[1] << 0, 4.5, 0;
  points[2] << 1, 1, 1;

  HyperPlane3d h(points);

  Eigen::Matrix<double, 1, N> A;
  double b;
  h.getKoordinateForm(A, b);

  Eigen::Matrix<double, 1, N> A_expected(1, 0.333333, 0.166667);
  double b_expected = 1.5;

  BOOST_TEST(b == b_expected);

  for (unsigned i = 0; i < N; ++i) {
    BOOST_TEST(A(0, i) == A_expected(0, i));
  }

  const HyperPlane3d::HyperSpacePoint t(1, 2);

  const HyperPlane3d::Point x = h(t);
  const auto t_ = h.inv(x);
  const auto x_ = h(t_);
  for (unsigned i = 0; i < N; ++i) {
    BOOST_TEST(x(i, 0) == x_(i, 0));
  }

  std::array<HyperPlane3d::HyperSpacePoint, N> ti;
  ti[0] << 1, 0;
  ti[1] << 0, 1;
  ti[2] << 0, 0;

  for (size_t i = 0; i < N; ++i) {
    const auto px = h(ti[i]);
    for (unsigned u = 0; u < N; ++u) {
      BOOST_TEST(px(u, 0) == points[i](u, 0));
    }
  }
}

BOOST_AUTO_TEST_CASE(test_Hyperplane_form_conversation_n10, *utf::tolerance(0.00001)) {

  constexpr unsigned N = 10;
  using HyperPlaneNd = opt::HyperPlane<N, double>;

  for (int run = 0; run < 100; ++run) {

    // get N random points

    std::array<HyperPlaneNd::Point, N> points;

    for (size_t p = 0; p < N; ++p) {
      points[p] = HyperPlaneNd::Point::Random();
    }

    HyperPlaneNd h(points);

    Eigen::Matrix<double, 1, N> A;
    double b;
    h.getKoordinateForm(A, b);

    // Test every point in points must be a solution for Ax = b
    for (size_t p = 0; p < N; ++p) {
      const double b_ = A * points[p];
      BOOST_TEST(b_ == b);
    }

    // Test t -> X and X -> t
    for (int sub_run = 0; sub_run < 1000; ++sub_run) {
      HyperPlaneNd::HyperSpacePoint t = HyperPlaneNd::HyperSpacePoint::Random().eval();
      const auto x = h(t);
      const auto t_ = h.inv(x);
      const auto x_ = h(t_);
      for (unsigned p = 0; p < N; ++p) {
        BOOST_TEST(x(p, 0) == x_(p, 0));
      }
      for (unsigned p = 0; p < N - 1; ++p) {
        BOOST_TEST(t(p, 0) == t_(p, 0));
      }
    }

    // Because we construct h from points in that order having t = e_i we should get the points we constructed h.
    for (unsigned i = 0; i < N - 1; ++i) {
      HyperPlaneNd::HyperSpacePoint te_i = HyperPlaneNd::HyperSpacePoint::Zero();
      te_i(i, 0) = 1.;
      const auto px = h(te_i);
      for (size_t p = 0; p < N; ++p) {
        BOOST_TEST(px(p, 0) == points[i](p, 0));
      }
    }
  }
}

#pragma clang diagnostic pop
