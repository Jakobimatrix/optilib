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
  using O = opt::ObjectiveType<P>;
  O obective{0};
  double score;

  constexpr unsigned BREAK_IF_FAULT = 100;
  constexpr unsigned MAX_STEPS = 10;

  opt::StoppingConditionMaxSteps<P> stop(MAX_STEPS);
  stop.init(&score, &obective);

  unsigned i = 0;
  while (!stop.applys() && i < BREAK_IF_FAULT) {
    stop.step();
    i++;
  }
  BOOST_TEST(i == MAX_STEPS);
}

BOOST_AUTO_TEST_CASE(test_StoppingConditionNStepsNoProgress) {

  constexpr unsigned P = 2;
  using O = opt::ObjectiveType<P>;
  O objective{0, 0};
  static const O obective_min_diff{0.1, 0.1};
  double score;

  constexpr unsigned BREAK_IF_FAULT = 100;
  constexpr unsigned MAX_STEPS = 10;
  constexpr unsigned NUM_STEPS_CANGE = 7;

  opt::StoppingConditionNStepsNoProgress<P> stop(MAX_STEPS, obective_min_diff);
  stop.init(&score, &objective);

  unsigned i = 0;
  while (!stop.applys() && i < BREAK_IF_FAULT) {
    if (i < NUM_STEPS_CANGE) {
      objective(0, 0) += 1;
    }
    stop.step();
    i++;
  }
  BOOST_TEST(i == MAX_STEPS + NUM_STEPS_CANGE);
}

BOOST_AUTO_TEST_CASE(test_StoppingConditionNStepsNoProgressNorm) {

  constexpr unsigned P = 2;
  using O = opt::ObjectiveType<P>;
  O obective{0, 0};
  double obective_min_diff = 0.1;

  constexpr unsigned BREAK_IF_FAULT = 100;
  constexpr unsigned MAX_STEPS = 10;
  constexpr unsigned NUM_STEPS_CANGE = 7;
  double score;

  opt::StoppingConditionNStepsNoProgressNorm<P> stop(MAX_STEPS, obective_min_diff);

  stop.init(&score, &obective);

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
  using O = opt::ObjectiveType<P>;
  O obective{0, 0};
  O dO{1, 1};
  static const O dO_thresh{0.1, 0.1};

  constexpr unsigned BREAK_IF_FAULT = 100;
  constexpr unsigned MAX_STEPS = 10;
  constexpr unsigned NUM_STEPS_CANGE = 7;
  double score;

  opt::StoppingConditionSmallDerivative<P> stop(MAX_STEPS, dO_thresh);
  stop.init(&score, &obective, &dO);

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

BOOST_AUTO_TEST_CASE(test_StoppingConditionTargetScore) {

  constexpr unsigned P = 2;
  using O = opt::ObjectiveType<P>;
  O obective{0, 0};
  O dO{1, 1};

  constexpr unsigned BREAK_IF_FAULT = 100;
  constexpr double TARGET_SCORE = 0.1;
  constexpr unsigned NUM_STEPS_CANGE = 7;

  double score = 100;

  opt::StoppingConditionTargetScore<P> stop(TARGET_SCORE);
  stop.init(&score, &obective);

  unsigned i = 0;
  while (!stop.applys() && i < BREAK_IF_FAULT) {
    i++;
    if (i >= NUM_STEPS_CANGE) {
      score = 0;
    }
    stop.step();
  }
  BOOST_TEST(i == NUM_STEPS_CANGE);
}

BOOST_AUTO_TEST_CASE(test_StoppingConditionMaxExecutionTime) {

  constexpr unsigned P = 2;
  using O = opt::ObjectiveType<P>;
  O obective{0, 0};
  O dO{1, 1};

  constexpr unsigned BREAK_IF_FAULT = 1000;
  constexpr unsigned long long MAX_EXEC_TIME_NS = 10'000'000;
  constexpr unsigned long long TASK_TIME = 1'000;

  double score = 100;

  int num_fails = 0;
  constexpr int NUM_ACCEPTED_FAILS_IN_100 = 1;
  // If the cpu pauses the optimizer in the wrong moment we will exceed the timelimit.
  // This might be less of a problem when running on a real time kernel.
  for (int l = 0; l < 100; ++l) {

    opt::StoppingConditionMaxExecutionTime<P> stop(MAX_EXEC_TIME_NS);
    stop.init(&score, &obective);

    unsigned i = 0;
    const auto start = opt::StoppingConditionMaxExecutionTime<P>::now();
    while (!stop.applys() && i < BREAK_IF_FAULT) {
      const auto task_start = opt::StoppingConditionMaxExecutionTime<P>::now();
      i++;
      bool task_finnished = false;
      while (!task_finnished) {
        const auto duration = opt::StoppingConditionMaxExecutionTime<P>::now() - task_start;
        task_finnished = duration >= TASK_TIME;
      }

      stop.step();
    }
    const auto end = opt::StoppingConditionMaxExecutionTime<P>::now();

    if (MAX_EXEC_TIME_NS < end - start) {
      num_fails++;
    }
  }
  BOOST_TEST(NUM_ACCEPTED_FAILS_IN_100 > num_fails);
}

#pragma clang diagnostic pop
