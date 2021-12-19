#define BOOST_TEST_MODULE stopping_test TestSuites
#define BOOST_TEST_DYN_LINK
#include <stdio.h>

#include <array>
#include <boost/test/unit_test.hpp>
#include <cmath>
#include <limits>
#include <numeric>
#include <optilib/stoppingcondition.hpp>

namespace utf = boost::unit_test;
namespace tt = boost::test_tools;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdisabled-macro-expansion"

BOOST_AUTO_TEST_CASE(test_StoppingConditionMaxSteps) {

  constexpr unsigned P = 1;
  using O = opt::Objective<P>;
  O obective{0};

  constexpr unsigned BREAK_IF_FAULT = 100;
  constexpr unsigned MAX_STEPS = 10;

  opt::StoppingConditionMaxSteps<P> stop(MAX_STEPS);
  stop.init(obective);

  unsigned i = 0;
  while (!stop.applys() && i < BREAK_IF_FAULT) {
    stop.step();
    i++;
  }
  BOOST_TEST(i == MAX_STEPS);
}

BOOST_AUTO_TEST_CASE(test_StoppingConditionNStepsNoProgress) {

  constexpr unsigned P = 2;
  using O = opt::Objective<P>;
  O obective{0, 0};
  O obective_min_diff{0.1, 0.1};

  constexpr unsigned BREAK_IF_FAULT = 100;
  constexpr unsigned MAX_STEPS = 10;
  constexpr unsigned NUM_STEPS_CANGE = 7;

  opt::StoppingConditionNStepsNoProgress<P> stop(MAX_STEPS, obective_min_diff);
  stop.init(obective);

  unsigned i = 0;
  while (!stop.applys() && i < BREAK_IF_FAULT) {
    if (i < NUM_STEPS_CANGE) {
      obective(0, 0) += 1;
    }
    stop.step();
    i++;
  }
  BOOST_TEST(i == MAX_STEPS + NUM_STEPS_CANGE);
}

BOOST_AUTO_TEST_CASE(test_StoppingConditionNStepsNoProgressNorm) {

  constexpr unsigned P = 2;
  using O = opt::Objective<P>;
  O obective{0, 0};
  double obective_min_diff = 0.1;

  constexpr unsigned BREAK_IF_FAULT = 100;
  constexpr unsigned MAX_STEPS = 10;
  constexpr unsigned NUM_STEPS_CANGE = 7;

  opt::StoppingConditionNStepsNoProgressNorm<P> stop(MAX_STEPS, obective_min_diff);

  stop.init(obective);

  unsigned i = 0;
  while (!stop.applys() && i < BREAK_IF_FAULT) {
    if (i < NUM_STEPS_CANGE) {
      obective(0, 0) += 1;
    }
    stop.step();
    i++;
  }
  BOOST_TEST(i == MAX_STEPS + NUM_STEPS_CANGE);
}

BOOST_AUTO_TEST_CASE(test_StoppingConditionSmallDerivative) {

  constexpr unsigned P = 2;
  using O = opt::Objective<P>;
  O obective{0, 0};
  O dO{1, 1};
  O dO_thresh{0.1, 0.1};

  constexpr unsigned BREAK_IF_FAULT = 100;
  constexpr unsigned MAX_STEPS = 10;
  constexpr unsigned NUM_STEPS_CANGE = 7;

  opt::StoppingConditionSmallDerivative<P> stop(MAX_STEPS, dO_thresh);
  stop.init(obective, dO);

  unsigned i = 0;
  while (!stop.applys() && i < BREAK_IF_FAULT) {
    if (i >= NUM_STEPS_CANGE) {
      dO = O::Zero();
    }
    stop.step();
    i++;
  }
  BOOST_TEST(i == MAX_STEPS + NUM_STEPS_CANGE);
}

#pragma clang diagnostic pop
