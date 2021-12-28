#define BOOST_TEST_MODULE sectioning_test TestSuites
#define BOOST_TEST_DYN_LINK
#include <stdio.h>

#include <array>
#include <boost/test/unit_test.hpp>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <optilib/sectioning.hpp>
#include <optilib/stoppingcondition.hpp>
namespace utf = boost::unit_test;
namespace tt = boost::test_tools;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdisabled-macro-expansion"

BOOST_AUTO_TEST_CASE(test_Sectioning_1d, *utf::tolerance(0.00001)) {
  constexpr size_t p = 1;
  using O = opt::Objective<p>;
  static const O MIN_PROGRESS{0.00001};

  const opt::ObjectiveFunction<1> parabular = [](const opt::Objective<1>& o) {
    return o(0, 0) * o(0, 0);
  };

  for (int ig = -100; ig < 100; ig += 5) {
    for (double sss = 0.01; sss < 100; sss *= 10) {
      const O initial_guess{ig};
      const O starting_stepsize{sss};

      auto stopping_cond_1 =
          std::make_shared<opt::StoppingConditionNStepsNoProgress<p>>(10u, MIN_PROGRESS);
      auto stopping_cond_2 = std::make_shared<opt::StoppingConditionMaxSteps<p>>(1000);

      opt::Sectioning<p> optimizer(parabular, initial_guess, starting_stepsize, true);
      optimizer.setStoppingCondition(stopping_cond_1);
      optimizer.setStoppingCondition(stopping_cond_2);

      optimizer.start();
      const auto optimum = optimizer.getCurrentOptimum();

      BOOST_TEST(optimum(0, 0) == 0);
    }
  }
}

#pragma clang diagnostic pop
