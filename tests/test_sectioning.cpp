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
  using O = opt::ObjectiveType<p>;
  static const O MIN_PROGRESS{0.00001};

  const opt::ObjectiveFunctionType<p> parabular = [](const O& o) {
    return o(0, 0) * o(0, 0);
  };

  for (int ig = -100; ig < 100; ig += 5) {
    for (double sss = 0.01; sss < 100; sss *= 10) {
      const O initial_guess{ig};
      const O starting_stepsize{sss};

      // these stopping conditions are rather lax so that the awkward cases
      // (bad initial guess and very small initial stepsize) also get to the minimum.
      auto stopping_cond_1 =
          std::make_shared<opt::StoppingConditionNStepsNoProgress<p>>(8u, MIN_PROGRESS);
      auto stopping_cond_2 = std::make_shared<opt::StoppingConditionMaxSteps<p>>(110);

      opt::Sectioning<p> optimizer(parabular, initial_guess, starting_stepsize, true);
      optimizer.setStoppingCondition(stopping_cond_1);
      optimizer.setStoppingCondition(stopping_cond_2);

      optimizer.start();
      const auto optimum = optimizer.getCurrentOptimum();

      BOOST_TEST(optimum(0, 0) == 0);
      return;
    }
  }
}

BOOST_AUTO_TEST_CASE(test_Sectioning_5d, *utf::tolerance(0.00001)) {

  constexpr size_t p = 5;
  using O = opt::ObjectiveType<p>;
  static const O MIN_PROGRESS =
      (O() << 0.00001, 0.00001, 0.00001, 0.00001, 0.00001).finished();

  const opt::ObjectiveFunctionType<p> parabular = [](const O& o) {
    return o(0, 0) * o(0, 0) + o(1, 0) * o(1, 0) + o(2, 0) * o(2, 0) +
           o(3, 0) * o(3, 0) + o(4, 0) * o(4, 0);
  };

  for (int ig = -100; ig < 100; ig += 5) {
    for (double sss = 0.01; sss < 100; sss *= 10) {
      const O initial_guess = (O() << ig, ig, ig, ig, ig).finished();
      const O starting_stepsize = (O() << sss, sss, sss, sss, sss).finished();

      // these stopping conditions are rather lax so that the awkward cases
      // (bad initial guess and very big/small initial stepsize) also get to the minimum.
      auto stopping_cond_1 =
          std::make_shared<opt::StoppingConditionNStepsNoProgress<p>>(24u, MIN_PROGRESS);
      auto stopping_cond_2 = std::make_shared<opt::StoppingConditionMaxSteps<p>>(600);

      opt::Sectioning<p> optimizer(parabular, initial_guess, starting_stepsize, true);
      optimizer.setStoppingCondition(stopping_cond_1);
      optimizer.setStoppingCondition(stopping_cond_2);

      optimizer.start();
      const auto optimum = optimizer.getCurrentOptimum();


      for (unsigned i = 0; i < p; ++i) {
        BOOST_TEST(optimum(i, 0) == 0);
      }
    }
  }
}

#pragma clang diagnostic pop
