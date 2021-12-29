#ifndef SECTIONING_HPP
#define SECTIONING_HPP

#include <cmath>

#include "optimizer.hpp"

/*
The Algorithm: Sectioning or sometimes taxi-cab is a grid-search.
It searches in one direction (1-d search) at a time by going along
one dimension in discete steps of the objective function until the
minimum of that 1-d search is reached. It then continues to search
along the next dimension. It stops when no step in any direction
improves the score.

To improve the basic Sectioning I added two new features:
1: Momentum
Momentum based gradient descent is a known methode to accellerate
the search and reduce the amount of needed points/calculations
on the objective function. The basic idea is to make the next "step"
longer if it goes in the same direction as the last step. Much like a ball
rolling faster downhills if its direction of decent does not change.
The momentum calculated here is just an analogy and has no physical meaning.

2. Reduce the step size by factor 10 each time the "normal sectioning" stops.
This allows to choose a rather big initial step size if the initial guess of the
optimum might be far from the real optimum.

Use when:
1. You dont know the gradient of the objective function
2. You did try Simplex-Downhill, Partikle Swarm and Gradient-Sectioning
and on average Sectioning performes better then thouse.
This might be rarely the case. One example where I found Sectioning
outperforming the other methods was the tuning of a PID controller. In
particular first optimizing P, than I than D (than repeat with smaller stepsize)
yielded in the best controllers following the defined step-response.
*/

namespace opt {
template <unsigned p, typename T = double, bool debug = false>
class Sectioning : public Optimizer<p, false, HESSIAN::NO, T> {
 public:
  using Objective = ObjectiveType<p, T>;
  using ObjectiveFunction = ObjectiveFunctionType<p, T>;

  /*!
   * \brief Sectioning Optimization.
   * \param objective_function The objective function to be minimized.
   * \param initial_guess The best guess of the location of the minima of the objective function.
   * \param starting_stepsize For each objective the maximal stepsize of the grid search.
   * \param improve_indefinitely If true the step size of the gridsearch will be lowered automatically.
   * In that case only given stopping conditions can stop the optimization process.
   */
  Sectioning(const ObjectiveFunction &objective_function,
             const Objective &initial_guess,
             const Objective &starting_stepsize,
             bool improve_indefinitely)
      : Optimizer<p, false, HESSIAN::NO, T>(
            [this]() { return step(); }, objective_function, initial_guess),
        stepsize(starting_stepsize),
        improve_indefinitely(improve_indefinitely) {
    reset();

    if constexpr (debug) {
      debugCurrentStep(getNextStep());
    }
  }

 private:
  /*!
   * \brief The sectioning algorithm.
   */
  bool step() noexcept {
    const auto step = getNextStep();
    const auto next_objective = this->getCurrentOptimum() + step;
    const T next_score = this->J(next_objective);
    if (this->getCurrentScore() <= next_score) {
      deccelerate();
      if (resetCondition()) {
        resetMomentum();
        if (!changeDirection()) {
          return false;
        }
      }
    } else {
      this->setNewOptimum(next_score, next_objective);
      accelerate();
      num_no_improvements = 0;
    }
    if constexpr (debug) {
      debugCurrentStep(step);
    }
    return true;
  }

  /*!
   * \brief Calculate the next direction to optimize.
   */
  bool changeDirection() noexcept {
    num_no_improvements++;
    if (num_no_improvements >= p * 2) {
      if (improve_indefinitely) {
        num_no_improvements = 0;
        stepsize *= 0.1;
      } else {
        return false;
      }
    }
    if (direction > 0) {
      direction = -1.;
    } else {
      direction = 1.;
      next_step_base(direction_index) = static_cast<T>(0.);
      direction_index++;
      if (direction_index >= p) {
        direction_index = 0u;
      }
    }
    next_step_base(direction_index) = direction * stepsize(direction_index, 0);
    return true;
  }

  /*!
   * \brief Reset internal momentum.
   */
  void resetMomentum() noexcept {
    acceleration = 0.1;
    momentum = 1.;
  }

  /*!
   * \brief Calaculate a new factor by which the step width gets multiplied for faster search in the same direction.
   */
  void accelerate() noexcept {
    momentum += acceleration;
    acceleration *= 2.;
  }

  /*!
   * \brief Calaculate a new smaller factor by which the step width gets multiplied.
   */
  void deccelerate() noexcept {
    int exp;
    std::frexp(momentum, &exp);
    const int b = static_cast<int>(exp > 2);
    const T reduce = static_cast<T>(b * exp + !b * 2);
    // Reduce the momentum rapidely and dynamicaly by divideing
    // the momentum by its exponent e (momentum = c*2^e).
    // At least divide by 2.
    momentum /= reduce;
    acceleration /= reduce;
  }

  /*!
   * \brief Calaculate the step size for the next step.
   */
  Objective getNextStep() const noexcept { return next_step_base * momentum; }

  /*!
   * \brief Condition at which we want to change search direction.
   */
  bool resetCondition() const noexcept { return momentum < 1.; }

  /*!
   * \brief Reinitialize the optimizer with default values.
   */
  void reset() {
    this->resetOptimizer();
    next_step_base = Objective::Zero();
    next_step_base(0, 0) = stepsize(0, 0);
    resetMomentum();
  }

  T direction = 1.;
  Objective stepsize;
  unsigned direction_index = 0u;

  unsigned num_no_improvements = 0;
  const bool improve_indefinitely;

  Objective next_step_base;
  T acceleration;
  T momentum;

 public:
  // debug
  struct DebugInfo {
    Objective objective;
    Objective step;
    T momentum;
    T score;
  };

  template <class Q = EnableType<debug>>
  typename std::enable_if<std::is_same<Q, TrueType>::value, const std::vector<DebugInfo> &>::type getDebugInfo() const
      noexcept {
    return debug_info;
  }

 private:
  template <class Q = EnableType<debug>>
  typename std::enable_if<std::is_same<Q, TrueType>::value, void>::type debugCurrentStep(
      const Objective &step) noexcept {
    debug_info.push_back(
        {this->getCurrentOptimum(), step, momentum, this->getCurrentScore()});
  }


  std::vector<DebugInfo> debug_info;
};
}  // namespace opt

#endif  // SECTIONING_HPP
