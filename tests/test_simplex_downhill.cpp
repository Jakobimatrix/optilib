#define BOOST_TEST_MODULE simplex_downhill_test TestSuites
#define BOOST_TEST_DYN_LINK
#include <stdio.h>

#include <array>
#include <boost/test/unit_test.hpp>
#include <cmath>
#include <limits>
#include <numeric>
#include <optilib/constraints.hpp>
#include <optilib/simplex_downhill.hpp>
#include <optilib/stoppingcondition.hpp>
namespace utf = boost::unit_test;
namespace tt = boost::test_tools;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdisabled-macro-expansion"

BOOST_AUTO_TEST_CASE(test_Sectioning_1d, *utf::tolerance(0.00001)) {

  return;


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

      opt::SimplexDownhill<p> optimizer(parabular, initial_guess, starting_stepsize);
      optimizer.setStoppingCondition(stopping_cond_1);
      optimizer.setStoppingCondition(stopping_cond_2);

      optimizer.start();
      const auto optimum = optimizer.getCurrentOptimum();

      BOOST_TEST(optimum(0, 0) == 0);
    }
  }
}

BOOST_AUTO_TEST_CASE(test_Sectioning_2d_linear_constraints_asymmetric,
                     *utf::tolerance(0.00001)) {
  constexpr size_t p = 2;
  using O = opt::ObjectiveType<p>;
  using LinearConstraint = opt::Constraint<p, true>;

  static const O MIN_PROGRESS{0.00001, 0.00001};
  static const O EXPECTED_OPTIMUM{1.49559, 3.56954};

  const opt::ObjectiveFunctionType<p> parabular = [](const O& o) {
    return o(0, 0) * o(0, 0) + o(1, 0) * o(1, 0);
  };

  double error = 0;
  O best;
  double score = 1000;
  for (int ig = 10; ig < 100; ig += 5) {
    for (double sss = 0.01; sss < 100; sss *= 10) {
      const O initial_guess{ig, ig};
      const O starting_stepsize{sss, sss};

      // these stopping conditions are rather lax so that the awkward cases
      // (bad initial guess and very small initial stepsize) also get to the minimum.
      auto stopping_cond_1 =
          std::make_shared<opt::StoppingConditionNStepsNoProgress<p>>(12u, MIN_PROGRESS);
      auto stopping_cond_2 = std::make_shared<opt::StoppingConditionMaxSteps<p>>(210);

      auto c1 = std::make_shared<LinearConstraint>(LinearConstraint::A(1, 2), 9.);

      opt::SimplexDownhill<p, double, true> optimizer(parabular, initial_guess, starting_stepsize);
      optimizer.setStoppingCondition(stopping_cond_1);
      optimizer.setStoppingCondition(stopping_cond_2);
      optimizer.setConstraint(c1);

      optimizer.start();
      const auto optimum = optimizer.getCurrentOptimum();

      if (score > optimizer.getCurrentScore() && optimum(0, 0) > 1) {
        score = optimizer.getCurrentScore();
        best = optimum;
      }
      error += (EXPECTED_OPTIMUM - optimum).norm();

      if ((EXPECTED_OPTIMUM - optimum).norm() > 1) {
        const auto& dbg_info = optimizer.getDebugInfo();
        const std::string name =
            "/tmp/" + std::to_string(ig) + "_" + std::to_string(sss) + ".csv";
        std::ofstream stream(name.c_str(), std::ofstream::trunc);
        opt::DebugInfo<p, double>::print(stream, ';', dbg_info);
      }


      BOOST_TEST(optimum(0, 0) == 1.79916);
      BOOST_TEST(optimum(1, 0) == 3.60042);
    }
  }
  std::cout << best.matrix();
  std::cout << "\n\n" << score;
  std::cout << "\n\n" << error;
}

BOOST_AUTO_TEST_CASE(test_Sectioning_2d_linear_constraints_symmetric,
                     *utf::tolerance(0.00001)) {

  return;

  constexpr size_t p = 2;
  using O = opt::ObjectiveType<p>;
  using LinearConstraint = opt::Constraint<p, true>;

  static const O MIN_PROGRESS{0.00001, 0.00001};

  const opt::ObjectiveFunctionType<p> parabular = [](const O& o) {
    return o(0, 0) * o(0, 0) + o(1, 0) * o(1, 0);
  };

  for (int ig = 10; ig < 100; ig += 5) {
    for (double sss = 0.01; sss < 100; sss *= 10) {
      const O initial_guess{ig, ig};
      const O starting_stepsize{sss, sss};

      // these stopping conditions are rather lax so that the awkward cases
      // (bad initial guess and very small initial stepsize) also get to the minimum.
      auto stopping_cond_1 =
          std::make_shared<opt::StoppingConditionNStepsNoProgress<p>>(8u, MIN_PROGRESS);
      auto stopping_cond_2 = std::make_shared<opt::StoppingConditionMaxSteps<p>>(110);

      auto c1 = std::make_shared<LinearConstraint>(LinearConstraint::A(1, 1), 9.);

      opt::SimplexDownhill<p, double> optimizer(parabular, initial_guess, starting_stepsize);
      optimizer.setStoppingCondition(stopping_cond_1);
      optimizer.setStoppingCondition(stopping_cond_2);
      optimizer.setConstraint(c1);

      optimizer.start();
      const auto optimum = optimizer.getCurrentOptimum();

      // const auto& dbg_info = optimizer.getDebugInfo();
      // std::ofstream stream("/tmp/aaaa.csv", std::ofstream::trunc);
      // opt::DebugInfo<p, double>::print(stream, ';', dbg_info);

      BOOST_TEST(optimum(0, 0) == 3);
      BOOST_TEST(optimum(1, 0) == 3);
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

      opt::SimplexDownhill<p> optimizer(parabular, initial_guess, starting_stepsize);
      // optimizer.setStoppingCondition(stopping_cond_1);
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
