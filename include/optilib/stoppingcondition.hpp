#ifndef OPTIMIZATION_STOPPINGCONDITION
#define OPTIMIZATION_STOPPINGCONDITION

#include <algorithm>
#include <chrono>

#include "definitions.hpp"

namespace opt {

// <DerivationRefHolder>
template <unsigned p, typename T, class Q, typename Enable = void>
struct DerivationRefHolder {};

template <unsigned p, typename T, class Q>
struct DerivationRefHolder<p, T, Q, std::enable_if_t<std::is_same<Q, TrueType>::value>> {
  using O = ObjectiveType<p, T>;
  const O *current_objective_derivative;
  const O &getDeviationOfCurrentObjective() const noexcept {
    return *current_objective_derivative;
  }
};

template <unsigned p, typename T, class Q>
struct DerivationRefHolder<p, T, Q, std::enable_if_t<std::is_same<Q, FalseType>::value>> {
};
// </DerivationRefHolder>

template <unsigned p, bool needs_derivative, typename T = double>
class StoppingCondition
    : public DerivationRefHolder<p, T, EnableType<needs_derivative>> {
 public:
  using O = ObjectiveType<p, T>;
  using dO = ObjectiveType<p, T>;

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
      const T *current_score_ptr, const O *current_objective_ptr, const O *current_objective_derivative_ptr) {
    current_objective = current_objective_ptr;
    current_score = current_score_ptr;
    this->current_objective_derivative = current_objective_derivative_ptr;
    reset();
  }

  template <class Q = EnableType<needs_derivative>>
  typename std::enable_if<std::is_same<Q, FalseType>::value, void>::type init(
      const T *current_score_ptr, const O *current_objective_ptr) {
    current_objective = current_objective_ptr;
    current_score = current_score_ptr;
    reset();
  }

  /*!
   * \brief Resets internal variables without changeing the desired stopping condition.
   */
  void reset() noexcept { resetInternal(); }

  /*!
   * \brief Reports if the stopping condition applys.
   * \return True if the stopping condition applys.
   */
  bool applys() const noexcept { return checkCondition(); }

  /*!
   * \brief Calculates the new condition. Needs to balled after every optimization step.
   */
  void step() noexcept { updateStep(); }

 protected:
  StoppingCondition(const std::function<bool()> &applys_callback,
                    const std::function<void()> &apply_internal_reset,
                    const std::function<void()> &apply_internal_step)
      : checkCondition(applys_callback),
        resetInternal(apply_internal_reset),
        updateStep(apply_internal_step) {}


  const O &getCurrentObjective() const noexcept { return *current_objective; }

  const O *current_objective;
  const T *current_score;

 private:
  const std::function<bool()> checkCondition = []() { return false; };
  const std::function<void()> resetInternal = []() {};
  const std::function<void()> updateStep = []() {};
};

template <unsigned p, typename T = double>
class StoppingConditionTargetScore : public StoppingCondition<p, false, T> {
 public:
  /*!
   * \brief Stop when score is below target
   * \param target_score The minimal target
   */
  StoppingConditionTargetScore(T target_score)
      : StoppingCondition<p, false, T>([this]() { return doesApply(); },
                                       [this]() { return doReset(); },
                                       [this]() { return doStep(); }),
        target_score(target_score) {}

 private:
  void doReset() noexcept { steps = 0; }

  bool doesApply() const noexcept {
    return target_score >= *this->current_score;
  }

  void doStep() noexcept {}

  unsigned steps = 0;
  const T target_score;
};

template <unsigned p, typename T = double>
class StoppingConditionMaxExecutionTime : public StoppingCondition<p, false, T> {
 public:
  /*!
   * \brief Stop when the time limit will be exceeded before the next
   * calculation.
   * \param max_execution_time_ns The time limit in nanoseconds.
   */
  StoppingConditionMaxExecutionTime(unsigned long long max_execution_time_ns)
      : StoppingCondition<p, false, T>([this]() { return doesApply(); },
                                       [this]() { return doReset(); },
                                       [this]() { return doStep(); }),
        max_execution_time_ns(max_execution_time_ns) {}

  unsigned long long static now() {
    timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return static_cast<unsigned long long>(now.tv_nsec) +
           static_cast<unsigned long long>(now.tv_sec) * 1'000'000'000ul;
  }

 private:
  void doReset() noexcept {
    unsigned long long average_step_time = 0;
    unsigned num_measurements = 0;
    start = now();
  }

  bool doesApply() const noexcept {
    const auto rightnow = now();
    const auto time_spent = rightnow - start;
    return max_execution_time_ns <= time_spent + average_step_time;
  }

  void doStep() noexcept {
    const auto rightnow = now();
    const auto time_spent = rightnow - start;
    average_step_time = time_spent / ++num_measurements;
  }

  unsigned long long average_step_time = 0;
  unsigned num_measurements = 0;
  unsigned long long start = now();
  const unsigned long long max_execution_time_ns;
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
      : StoppingCondition<p, false, T>([this]() { return doesApply(); },
                                       [this]() { return doReset(); },
                                       [this]() { return doStep(); }),
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
  using O = ObjectiveType<p>;

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
      : StoppingCondition<p, false, T>([this]() { return doesApply(); },
                                       [this]() { return doReset(); },
                                       [this]() { return doStep(); }),
        objective_delta(min_objective_delta.cwiseAbs()),
        max_steps_no_progress(std::max(max_steps_no_progress, 1u)) {}

 private:
  void doReset() noexcept {
    steps = 0;
    last_objective = this->getCurrentObjective();
  }

  bool doesApply() const noexcept { return steps >= max_steps_no_progress; }


  void doStep() noexcept {
    steps++;
    const O diff = last_objective - this->getCurrentObjective();
    for (size_t i = 0; i < p; ++i) {
      if (std::abs(diff(i, 0)) > objective_delta(i, 0)) {
        steps = 0;
        last_objective = this->getCurrentObjective();
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
  using O = ObjectiveType<p>;

  /*!
   * \brief Stop if after some steps the objectiv did not became smaller.
   * ! For the criteria to activate the norm of all objectives did not decreased
   * for more than min_objective_delta for max_steps_no_progress steps !
   * \param max_steps_no_progress The maximal number of iterations the optimizer
   * is allowed to do without progress.
   * \param min_objective_delta The minimal norm accepted as progress.
   */
  StoppingConditionNStepsNoProgressNorm(unsigned max_steps_no_progress, double min_objective_delta)
      : StoppingCondition<p, false, T>([this]() { return doesApply(); },
                                       [this]() { return doReset(); },
                                       [this]() { return doStep(); }),
        objective_delta(std::abs(min_objective_delta)),
        max_steps_no_progress(std::max(max_steps_no_progress, 1u)) {}

 private:
  void doReset() noexcept {
    steps = 0;
    last_objective = this->getCurrentObjective();
  }

  bool doesApply() const noexcept { return steps >= max_steps_no_progress; }

  void doStep() noexcept {

    O diff = last_objective - this->getCurrentObjective();
    const bool no_progress = diff.norm() < objective_delta;

    if (no_progress) {
      steps++;
    } else {
      steps = 0;
      last_objective = this->getCurrentObjective();
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
  using dO = ObjectiveType<p, T>;

  /*!
   * \brief Stop when derivative is smaller for n steps than given threshold.
   * to do.
   */
  StoppingConditionSmallDerivative(unsigned max_steps, dO threshold_derivative)
      : StoppingCondition<p, true, T>([this]() { return doesApply(); },
                                      [this]() { return doReset(); },
                                      [this]() { return doStep(); }),
        max_steps(std::max(max_steps, 1u)),
        threshold(threshold_derivative.cwiseAbs()) {}

 private:
  void doReset() noexcept { steps = 0; }

  bool doesApply() const noexcept { return steps >= max_steps; }

  void doStep() noexcept {
    steps++;
    for (size_t i = 0; i < p; ++i) {
      if (threshold(i, 0) < this->getDeviationOfCurrentObjective()(i, 0)) {
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
