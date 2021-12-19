#ifndef SECTIONING_HPP
#define SECTIONING_HPP

#include <limits>

#include "optimizer.hpp"

namespace opt {
template <unsigned p, typename T = double, bool enable_step_record = false>
class Sectioning : public Optimizer<p, false, T, enable_step_record> {
 public:
  using O = Objective<p, T>;
  using J = ObjectiveFunction<p, T>;

  Sectioning(const J &objective_function, O initial_guess, const O &starting_stepsize)
      : Optimizer<p, false, T, enable_step_record>(
            std::bind(&Sectioning<p, T, enable_step_record>::step, this)),
        stepsize(starting_stepsize),
        objective_function(objective_function) {
    this->current_objective = initial_guess;
    this->current_score = objective_function(this->current_objective);
    reset();
  }

 private:
  bool step() noexcept {
    prepareNextStep();
    const O last_objective = this->current_objective;
    const T last_score = current_score;
    this->current_objective += next_step;
    current_score = objective_function(this->current_objective);

    if (current_score < last_score) {
      current_score = last_score;
      this->current_objective = last_objective;
      deccelerate();
      if (resetCondition()) {
        resetMomentum();
        changeDirection();
      }
    } else {
      accelerate();
      num_no_improvements = 0;
    }
    return true;
  }

  void changeDirection() noexcept {
    num_no_improvements++;
    if (num_no_improvements >= p * 2) {
      num_no_improvements = 0;
      stepsize *= 0.1;
    }
    if (direction > 0) {
      direction = -1.;
    } else {
      direction = 1.;
      direction_index++;
      if (direction_index > p) {
        direction_index = 0u;
      }
    }
    next_step = O::Zero();
    next_step(direction_index) = direction * stepsize(direction_index, 0);
  }

  void resetMomentum() noexcept {
    acceleration = 0.3;
    momentum = 1;
  }

  void accelerate() noexcept {
    momentum += acceleration;
    acceleration *= 2.;
  }

  void prepareNextStep() noexcept { next_step(direction_index, 0) *= momentum; }

  void deccelerate() noexcept {
    momentum / 2.;
    acceleration /= 2.;
  }

  bool resetCondition() const noexcept { return momentum < 1; }

  void reset() { next_step(0, 0) = stepsize(0, 0); }

  T direction = 1.;
  O stepsize;
  unsigned direction_index = 0u;
  T current_score = std::numeric_limits<T>::max();

  unsigned num_no_improvements = 0;

  O next_step = O::Zero();
  T acceleration = 0.3;
  T momentum = 1;
  J objective_function;
};
}  // namespace opt

#endif  // SECTIONING_HPP
