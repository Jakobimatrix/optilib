#include <fstream>
#include <iostream>
#include <optilib/gradient_sectioning.hpp>
#include <optilib/sectioning.hpp>
#include <optilib/simplex_downhill.hpp>

/*
int main() {


  constexpr size_t p = 2;
  using O = opt::ObjectiveType<p>;
  O MIN_PROGRESS;
  MIN_PROGRESS.fill(0.00001);

  const opt::ObjectiveFunctionType<p> parabular = [](const O& o) {
    double ret = 0;
    for (unsigned i = 0; i < p; ++i) {
      ret += o(i, 0) * o(i, 0);
    }
    return ret;
  };

  size_t average_grad = 0;
  size_t average_sec = 0;
  size_t i = 0;
  for (int ig = -100; ig < 100; ig += 5) {
    i++;
    std::ofstream file1, file2;
    const std::string filename1 =
        "/tmp/gradientSectioning_" + std::to_string(ig) + ".csv";
    const std::string filename2 =
        "/tmp/Sectioning_" + std::to_string(ig) + ".csv";
    file1.open(filename1.c_str(), std::ios::out | std::ios::trunc);
    file2.open(filename2.c_str(), std::ios::out | std::ios::trunc);


    double sss = 1;
    O initial_guess;
    initial_guess.fill(ig);
    O starting_stepsize;
    starting_stepsize.fill(sss);

    auto stopping_cond_1 =
        std::make_shared<opt::StoppingConditionNStepsNoProgress<p>>(6u, MIN_PROGRESS);
    auto stopping_cond_2 = std::make_shared<opt::StoppingConditionMaxSteps<p>>(100);
    auto stopping_cond_3 = std::make_shared<opt::StoppingConditionTargetScore<p>>(0.00001);

    opt::GradientSectioning<p, double, true> optimizer1(
        parabular, initial_guess, starting_stepsize);
    optimizer1.setStoppingCondition(stopping_cond_1);
    optimizer1.setStoppingCondition(stopping_cond_2);
    optimizer1.setStoppingCondition(stopping_cond_3);
    optimizer1.start();
    const auto optimum1 = optimizer1.getCurrentOptimum();
    const auto& info1 = optimizer1.getDebugInfo();


    opt::Sectioning<p, double, true> optimizer2(
        parabular, initial_guess, starting_stepsize, true);
    optimizer2.setStoppingCondition(stopping_cond_1);
    optimizer2.setStoppingCondition(stopping_cond_2);
    optimizer2.setStoppingCondition(stopping_cond_3);
    optimizer2.start();
    const auto optimum2 = optimizer2.getCurrentOptimum();
    const auto& info2 = optimizer2.getDebugInfo();

    opt::DebugInfo<p>::print(file1, ';', info1);
    opt::DebugInfo<p>::print(file2, ';', info2);


    average_grad += info1.size();
    average_sec += info2.size();
    std::cout << "------------------\n ig:" << initial_guess.transpose().matrix()
              << "\n sss: " << starting_stepsize.transpose().matrix() << "\n"
              << "sectioning [" << info2.size()
              << "]: opt: " << optimum2.transpose().matrix() << "\n"
              << "G sectioning[" << info1.size()
              << "]: opt: " << optimum1.transpose().matrix() << "\n";
  }

  std::cout << "\naverage grad: " << average_grad / i
            << "\naverage sec: " << average_sec / i << "\n";
  return 0;
}
*/

int main() {

  constexpr size_t p = 2;
  using O = opt::ObjectiveType<p>;
  using LinearConstraint = opt::Constraint<p, true>;

  static const O MIN_PROGRESS{0.00001, 0.00001};
  static const O EXPECTED_OPTIMUM{1.79916, 3.60042};

  const opt::ObjectiveFunctionType<p> parabular = [](const O& o) {
    return o(0, 0) * o(0, 0) + o(1, 0) * o(1, 0);
  };


  int ig = 75;
  double sss = 10;

  const O initial_guess{ig, ig};
  const O starting_stepsize{sss, sss};

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


  const auto& dbg_info = optimizer.getDebugInfo();
  const std::string name =
      "/tmp/" + std::to_string(ig) + "_" + std::to_string(sss) + ".csv";
  std::ofstream stream(name.c_str(), std::ofstream::trunc);
  opt::DebugInfo<p, double>::print(stream, ';', dbg_info);


  const std::string name2 =
      "/tmp/" + std::to_string(ig) + "_" + std::to_string(sss) + "2.csv";
  std::ofstream stream2(name2.c_str(), std::ofstream::trunc);

  const auto& simplxe = optimizer.simplexe;
  const char SEPERATOR = ';';
  for (const auto& pair : simplxe) {
    const auto& vs = pair.second.vertices;
    for (const auto& v : vs) {
      stream2 << v.second(0, 0) << SEPERATOR << v.second(1, 0) << SEPERATOR
              << v.first << SEPERATOR;
    }
    stream2 << pair.first << '\n';
  }
}
