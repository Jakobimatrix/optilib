#ifndef OPTIMIZATION_STOPPINGCONDITION
#define OPTIMIZATION_STOPPINGCONDITION

#include <algorithm>
#include <iostream>

#include "definitions.hpp"

namespace opt {

// <DerivationRefHolder>
template <unsigned p, typename T, class Q, typename Enable = void>
struct DerivationRefHolder {};

template <unsigned p, typename T, class Q>
struct DerivationRefHolder<p, T, Q, std::enable_if_t<std::is_same<Q, TrueType>::value>> {
  const Objective<p, T> *current_objective_derivative;
};

template <unsigned p, typename T, class Q>
struct DerivationRefHolder<p, T, Q, std::enable_if_t<std::is_same<Q, FalseType>::value>> {

  DerivationRefHolder(){};
};
// </DerivationRefHolder>

template <unsigned p, bool needs_derivative, typename T = double>
class StoppingCondition
    : public DerivationRefHolder<p, T, EnableType<needs_derivative>> {
 public:
  using O = Objective<p, T>;
  using dO = Objective<p, T>;

  StoppingCondition(const StoppingCondition &other) = default;
  StoppingCondition(StoppingCondition &&other) = default;
  StoppingCondition &operator=(const StoppingCondition &other) = default;
  StoppingCondition &operator=(StoppingCondition &&other) = default;

  /*!
   * \brief Before using the stopping condition it must be initiated.
   * \param objective Reference to the current best solution of the optimizer.
   * \param objective_derivative eference to the current best solutions
   * derivative of the optimizer.
   */

  template <class Q = EnableType<needs_derivative>>
  typename std::enable_if<std::is_same<Q, TrueType>::value, void>::type init(
      const O *objective, const dO *objective_derivative) {
    current_objective = objective;
    this->current_objective_derivative = objective_derivative;
    initInternal();
    reset();
  }

  // template <class Q = EnableType<needs_derivative>>
  // typename std::enable_if<std::is_same<Q, FalseType>::value, void>::type
  void init(const O *objective) {
    static_assert(!needs_derivative,
                  "The choosen Stopping condition needs a derivative to work.");
    current_objective = objective;
    initInternal();
    reset();
  }

  void reset() noexcept { resetInternal(); }

  bool applys() const noexcept { return checkCondition(); }

  void step() noexcept { updateStep(); }

 protected:
  StoppingCondition(const std::function<bool()> &applys_callback,
                    const std::function<void()> &apply_internal_reset,
                    const std::function<void()> &apply_internal_init,
                    const std::function<void()> &apply_internal_update)
      : checkCondition(applys_callback),
        resetInternal(apply_internal_reset),
        updateStep(apply_internal_update),
        initInternal(apply_internal_init) {}

  StoppingCondition(const std::function<bool()> &applys_callback,
                    const std::function<void()> &apply_internal_reset,
                    const std::function<void()> &apply_internal_update)
      : checkCondition(applys_callback),
        resetInternal(apply_internal_reset),
        updateStep(apply_internal_update) {}

  const O *current_objective;

  const std::function<bool()> checkCondition = []() { return false; };
  const std::function<void()> resetInternal = []() {};
  const std::function<void()> updateStep = []() {};
  const std::function<void()> initInternal = []() {};
};

template <unsigned p, typename T = double>
class StoppingConditionMaxSteps : public StoppingCondition<p, false, T> {
 public:
  /*!
   * \brief Stop when max steps reached
   * \param max_steps The maximal number of iterations the optimizer is allowed
   * to do.
   */
  StoppingConditionMaxSteps(unsigned max_steps)
      : StoppingCondition<p, false, T>(
            std::bind(&StoppingConditionMaxSteps<p, T>::doesApply, this),
            std::bind(&StoppingConditionMaxSteps<p, T>::doReset, this),
            std::bind(&StoppingConditionMaxSteps<p, T>::doStep, this)),
        max_steps(max_steps) {}

 private:
  void doReset() noexcept { steps = 0; }

  bool doesApply() const noexcept { return steps >= max_steps; }

  void doStep() noexcept { ++steps; }

  unsigned steps = 0;
  const unsigned max_steps;
};

template <unsigned p, typename T = double>
class StoppingConditionNStepsNoProgress : public StoppingCondition<p, false, T> {
 public:
  using O = Objective<p>;

  /*!
   * \brief Stop if after some steps the objectiv did not became smaller.
   * ! For the criteria to activate no objective decreased for more than
   * min_objective_delta for max_steps_no_progress steps !
   * \param max_steps_no_progress The maximal number of iterations the optimizer
   * is allowed to do without progress.
   * \param min_objective_delta For each objective the minimal (positive) delta
   * which counts as progress.
   */
  StoppingConditionNStepsNoProgress(unsigned max_steps_no_progress, O min_objective_delta)
      : StoppingCondition<p, false, T>(
            std::bind(&StoppingConditionNStepsNoProgress<p, T>::doesApply, this),
            std::bind(&StoppingConditionNStepsNoProgress<p, T>::doReset, this),
            std::bind(&StoppingConditionNStepsNoProgress<p, T>::doInit, this),
            std::bind(&StoppingConditionNStepsNoProgress<p, T>::doStep, this)),
        objective_delta(min_objective_delta.cwiseAbs()),
        max_steps_no_progress(std::max(max_steps_no_progress, 1u)) {}

 private:
  void doReset() noexcept { steps = 0; }

  bool doesApply() const noexcept { return steps >= max_steps_no_progress; }

  void doInit() noexcept { last_objective = *this->current_objective; }

  void doStep() noexcept {
    steps++;
    const O diff = last_objective - *this->current_objective;
    for (size_t i = 0; i < p; ++i) {
      if (std::abs(diff(i, 0)) > objective_delta(i, 0)) {
        steps = 0;
        last_objective = *this->current_objective;
        break;
      }
    }
  }

  unsigned steps = 0;
  O last_objective;
  const O objective_delta;
  const unsigned max_steps_no_progress;
};

template <unsigned p, typename T = double>
class StoppingConditionNStepsNoProgressNorm : public StoppingCondition<p, false, T> {
 public:
  using O = Objective<p>;

  /*!
   * \brief Stop if after some steps the objectiv did not became smaller.
   * ! For the criteria to activate the norm of all objectives did not decreased
   * for more than min_objective_delta for max_steps_no_progress steps !
   * \param max_steps_no_progress The maximal number of iterations the optimizer
   * is allowed to do without progress.
   * \param min_objective_delta The minimal norm accepted as progress.
   */
  StoppingConditionNStepsNoProgressNorm(unsigned max_steps_no_progress, double min_objective_delta)
      : StoppingCondition<p, false, T>(
            std::bind(&StoppingConditionNStepsNoProgressNorm<p, T>::doesApply, this),
            std::bind(&StoppingConditionNStepsNoProgressNorm<p, T>::doReset, this),
            std::bind(&StoppingConditionNStepsNoProgressNorm<p, T>::doInit, this),
            std::bind(&StoppingConditionNStepsNoProgressNorm<p, T>::doStep, this)),
        objective_delta(std::abs(min_objective_delta)),
        max_steps_no_progress(std::max(max_steps_no_progress, 1u)) {}

 private:
  void doReset() noexcept { steps = 0; }

  bool doesApply() const noexcept { return steps >= max_steps_no_progress; }

  void doInit() noexcept { last_objective = *this->current_objective; }

  void doStep() noexcept {

    O diff = last_objective - *this->current_objective;
    const bool no_progress = diff.norm() < objective_delta;

    if (no_progress) {
      steps++;
    } else {
      steps = 0;
      last_objective = *this->current_objective;
    }
  }

  unsigned steps = 0;
  O last_objective;
  const double objective_delta;
  const unsigned max_steps_no_progress;
};

template <unsigned p, typename T = double>
class StoppingConditionSmallDerivative : public StoppingCondition<p, true, T> {
 public:
  using dO = Objective<p, T>;

  /*!
   * \brief Stop when derivative is smaller for n steps than given threshold.
   * to do.
   */
  StoppingConditionSmallDerivative(unsigned max_steps, dO threshold_derivative)
      : StoppingCondition<p, true, T>(
            std::bind(&StoppingConditionSmallDerivative<p, T>::doesApply, this),
            std::bind(&StoppingConditionSmallDerivative<p, T>::doReset, this),
            std::bind(&StoppingConditionSmallDerivative<p, T>::doStep, this)),
        max_steps(std::max(max_steps, 1u)),
        threshold(threshold_derivative.cwiseAbs()) {}

 private:
  void doReset() noexcept { steps = 0; }

  bool doesApply() const noexcept { return steps >= max_steps; }

  void doStep() noexcept {
    steps++;
    for (size_t i = 0; i < p; ++i) {
      if (threshold(i, 0) < (*this->current_objective_derivative)(i, 0)) {
        steps = 0;
        break;
      }
    }
  }

  unsigned steps = 0;
  const unsigned max_steps;
  dO threshold;
};

}  // namespace opt

#endif
