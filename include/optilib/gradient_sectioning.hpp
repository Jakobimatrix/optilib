#ifndef GRADIENT_SECTIONING_HPP
#define GRADIENT_SECTIONING_HPP

#include <cmath>

#include "optimizer.hpp"

/*
The Algorithm: Gradient-Sectioning Is very simmilar to Sectioning (see sectioning.hpp).
The only difference: I approximate the steepest decend and follow that
direction until on that line the minima is reached. Then I approximate the
next steepest decent from there. It is momentum based and also reduces the
stepsize if necessary.

Advantage against Sectioning:
It will find the optimum in vewer steps especially on smooth objective functions.

Disadvantage against Sectioning:
Every time it changes direction I need to calculate 2*(2^p - 1) probes on the
objective function and the corresponding score. If the objective function is heavy to
compute or the objective function is rather staircase-shaped, the normal Sectioning
might be faster.

Use when:
1. You dont know the gradient of the objective function
2. You did try Simplex-Downhill (and Partikle Swarm)
and on average Gradient-Sectioning performes better then thouse.
*/

namespace opt {
template <unsigned p, typename T = double, bool debug = false>
class GradientSectioning : public Optimizer<p, false, HESSIAN::NO, T, debug> {
 public:
  using Objective = ObjectiveType<p, T>;
  using ObjectiveFunction = ObjectiveFunctionType<p, T>;

  /*!
   * \brief Gradient-Sectioning Optimization.
   * \param objective_function The objective function to be minimized.
   * \param initial_guess The best guess of the location of the minima of the objective function.
   * \param starting_stepsize For each objective the maximal stepsize of the grid search.
   * \param improve_indefinitely If true the step size of the gridsearch will be lowered automatically.
   * In that case only given stopping conditions can stop the optimization process.
   */
  GradientSectioning(const ObjectiveFunction &objective_function,
                     const Objective &initial_guess,
                     const Objective &starting_stepsize)
      : Optimizer<p, false, HESSIAN::NO, T, debug>(
            [this]() { return step(); }, objective_function, initial_guess),
        stepsize(starting_stepsize) {
    reset();
  }

 private:
  /*!
   * \brief The sectioning algorithm.
   */
  bool step() noexcept {
    if (!has_new_direction) {
      changeDirection();
      return true;
    }
    const auto step = getNextStep();
    const auto next_objective = this->getCurrentOptimum() + step;
    const T next_score = this->J(next_objective);
    if (this->getCurrentScore() <= next_score) {
      this->debugCurrentStep(next_objective, next_score);
      deccelerate();
      if (resetCondition()) {
        resetMomentum();
        changeDirection();
      }
    } else {
      this->setNewOptimum(next_score, next_objective);
      accelerate();
    }
    return true;
  }

  /*!
   * \brief Calculate the next direction to optimize.
   */
  void changeDirection() noexcept {
    if (num_no_improvements < 2 && has_new_direction) {
      next_step_base = -next_step_base;
      return;
    }
    num_no_improvements = 0;

    Objective best_direction = Objective::Zero();
    T best_score = this->getCurrentScore();

    constexpr T one = static_cast<T>(1);
    constexpr T zero = static_cast<T>(0);

    Objective current_direction = Objective::Zero();
    has_new_direction = false;
    constexpr auto num_different_directions = ipow(2, p) - 1;
    for (int i = 0; i < num_different_directions; ++i) {
      T b = static_cast<T>(current_direction(0, 0) + 1 > 1);
      auto carry = b * one + !b * zero;
      current_direction(0, 0) = one - carry;
      for (size_t j = 1; j < p; ++j) {
        current_direction(j, 0) += carry;
        b = static_cast<T>(current_direction(j, 0) > 1);
        current_direction(j, 0) = b * zero + !b * current_direction(j, 0);
        carry = b * one + !b * zero;
      }

      const Objective direction = current_direction.cwiseProduct(stepsize);
      const Objective probe1 = this->getCurrentOptimum() + direction;
      const Objective probe2 = this->getCurrentOptimum() - direction;

      const T score1 = this->J(probe1);
      const T score2 = this->J(probe2);

      if constexpr (debug) {
        this->debugCurrentStep(probe1, score1);
        this->debugCurrentStep(probe2, score2);
      }

      const bool is_probe1 = score1 < score2;
      const T score = is_probe1 ? score1 : score2;
      const Objective probe = is_probe1 ? direction : -direction;

      if (score < best_score) {
        best_direction = probe;
        best_score = score;
        has_new_direction = true;
      }
    }

    if (has_new_direction) {
      next_step_base = best_direction;
      this->setNewOptimum(best_score, this->getCurrentOptimum() + best_direction);
      accelerate();
    }
    stepsize *= 0.1;
  }

  /*!
   * \brief Reset internal momentum.
   */
  void resetMomentum() noexcept {
    acceleration = 0.1;
    momentum = 1.;
    num_no_improvements++;
  }

  /*!
   * \brief Calaculate a new factor by which the step width gets multiplied for faster search in the same direction.
   */
  void accelerate() noexcept {
    momentum += acceleration;
    acceleration *= 2.;
    num_no_improvements = 0;
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
    has_new_direction = false;
    direction = 1.;
    num_no_improvements = 0;
    resetMomentum();
  }

  Objective stepsize;
  unsigned direction_index = 0u;

  T direction;
  unsigned num_no_improvements;
  bool has_new_direction;

  Objective next_step_base;
  T acceleration;
  T momentum;
};
}  // namespace opt

#endif  // GRADIENT_SECTIONING_HPP
