#include <iostream>
#include <optilib/gradient_sectioning.hpp>
#include <optilib/sectioning.hpp>

int main() {

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

      auto stopping_cond_1 =
          std::make_shared<opt::StoppingConditionNStepsNoProgress<p>>(24u, MIN_PROGRESS);
      auto stopping_cond_2 = std::make_shared<opt::StoppingConditionMaxSteps<p>>(600);

      opt::GradientSectioning<p, double, true> optimizer1(
          parabular, initial_guess, starting_stepsize, true);
      optimizer1.setStoppingCondition(stopping_cond_1);
      optimizer1.setStoppingCondition(stopping_cond_2);
      optimizer1.start();
      const auto optimum1 = optimizer1.getCurrentOptimum();
      const auto& info1 = optimizer1.getDebugInfo();


      opt::Sectioning<p, double, true> optimizer2(
          parabular, initial_guess, starting_stepsize, true);
      optimizer2.setStoppingCondition(stopping_cond_1);
      optimizer2.setStoppingCondition(stopping_cond_2);
      optimizer2.start();
      const auto optimum2 = optimizer2.getCurrentOptimum();
      const auto& info2 = optimizer2.getDebugInfo();

      std::cout << "------------------\n ig:" << initial_guess.transpose().matrix()
                << "\n sss: " << starting_stepsize.transpose().matrix() << "\n"
                << "sectioning [" << info2.size()
                << "]: opt: " << optimum2.transpose().matrix() << "\n"
                << "G sectioning[" << info1.size()
                << "]: opt: " << optimum1.transpose().matrix() << "\n";
    }
  }
  return 0;
}
