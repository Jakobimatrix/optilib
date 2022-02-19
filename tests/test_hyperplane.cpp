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

BOOST_AUTO_TEST_CASE(test_Hyperplane_form_conversation, *utf::tolerance(0.00001)) {

  using HyperPlane3d = opt::HyperPlane<3, double>;

  std::array<HyperPlane3d::Point, 3> points;
  points[0] << 0, 0, 9;
  points[1] << 0, 4.5, 0;
  points[2] << 1, 1, 1;

  HyperPlane3d h(points);

  Eigen::Matrix<double, 1, 3> A;
  double b;
  h.getKoordinateForm(A, b);

  Eigen::Matrix<double, 1, 3> A_expected(1, 0.333333, 0.166667);
  double b_expected = 1.5;

  BOOST_TEST(b == b_expected);

  for (unsigned i = 0; i < 3; ++i) {
    BOOST_TEST(A(0, i) == A_expected(0, i));
  }

  const HyperPlane3d::HyperSpacePoint t(1, 2);

  const HyperPlane3d::Point x = h(t);
  const auto t_ = h.inv(x);

  for (unsigned i = 0; i < 2; ++i) {
    BOOST_TEST(t(i, 0) == t_(i, 0));
  }

  std::array<HyperPlane3d::HyperSpacePoint, 3> ti;
  ti[0] << 1, 0;
  ti[1] << 0, 1;
  ti[2] << 0, 0;

  for (size_t i = 0; i < 3; ++i) {
    const auto px = h(ti[i]);
    for (unsigned u = 0; u < 3; ++u) {
      BOOST_TEST(px(u, 0) == points[i](u, 0));
    }
  }
}


#pragma clang diagnostic pop
