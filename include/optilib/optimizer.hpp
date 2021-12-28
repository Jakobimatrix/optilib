#ifndef OPTIMIZATION_OPTIMIZE
#define OPTIMIZATION_OPTIMIZE

#include <memory>
#include <vector>

#include "definitions.hpp"
#include "stoppingcondition.hpp"

namespace opt {

// <DerivationHolder>
template <unsigned p, typename T, class Q, typename Enable = void>
struct DerivationHolder {};

template <unsigned p, typename T, class Q>
struct DerivationHolder<p, T, Q, std::enable_if_t<std::is_same<Q, TrueType>::value>> {
  using dO = Objective<p, T>;
  dO current_objective_derivative;
};

template <unsigned p, typename T, class Q>
class DerivationHolder<p, T, Q, std::enable_if_t<std::is_same<Q, FalseType>::value>> {};
// </DerivationHolder>

// <RecordSteps>
template <unsigned p, typename T, class Q, typename Enable = void>
class RecordSteps {};

template <unsigned p, typename T, class Q>
class RecordSteps<p, T, Q, std::enable_if_t<std::is_same<Q, TrueType>::value>> {
  using O = Objective<p, T>;

 public:
  const std::vector<O> &getRecordSteps() const noexcept { return steps; }

  void recordStep(const O &current_objective) noexcept {
    steps.push_back(current_objective);
  }

 protected:
  std::vector<O> steps;
};

template <unsigned p, typename T, class Q>
struct RecordSteps<p, T, Q, std::enable_if_t<std::is_same<Q, FalseType>::value>> {};
// </RecordSteps>

template <unsigned p, bool has_derivative, typename T = double, bool enable_step_record = false>
class Optimizer : public DerivationHolder<p, T, EnableType<enable_step_record>>,
                  public RecordSteps<p, T, EnableType<enable_step_record>> {
 public:
  using O = Objective<p, T>;
  using dO = Objective<p, T>;
  using J = ObjectiveFunction<p, T>;
  using dJdO = ObjectiveFunctionDeviation<p, T>;

  O getCurrentOptimum() noexcept { return current_objective; }
  O &getCurrentOptimum() const noexcept { return current_objective; }

  template <class Q = EnableType<has_derivative>>
  typename std::enable_if<std::is_same<Q, TrueType>::value, void>::type setStoppingCondition(
      std::shared_ptr<StoppingCondition<p, true, T>> c) {
    if (c == nullptr) {
      return;
    }
    c->init(&current_objective, &this->current_objective_derivation);
    stopping_conditions_with_derivative.push_back(c);
  }

  // template <class Q = EnableType<has_derivative>>
  // typename std::enable_if<std::is_same<Q, FalseType>::value, void>::type
  void setStoppingCondition(std::shared_ptr<StoppingCondition<p, false, T>> c) {
    static_assert(!has_derivative,
                  "The Choosen optimizer has no access to the derivative of "
                  "the Objective function and thus can not stop on a stopping "
                  "condition which needs the derivative.");
    if (c == nullptr) {
      return;
    }
    c->init(&current_objective);
    stopping_conditions.push_back(c);
  }

  void start() noexcept {
    if (!step()) {
      return;
    }
    do {
      if constexpr (enable_step_record) {
        this->recordStep(current_objective);
      }

      updateStoppingConditions();
    } while (!stop() && step());
  }

 protected:
  Optimizer(const std::function<bool()> &step_function) : step(step_function){};

  bool stop() {
    for (const auto &c : stopping_conditions) {
      if (c->applys()) {
        return true;
      }
    }
    if constexpr (has_derivative) {
      for (auto &c : stopping_conditions_with_derivative) {
        if (c->applys()) {
          return true;
        }
      }
    }
    return false;
  }

  void updateStoppingConditions() {
    for (auto &c : stopping_conditions) {
      c->step();
    }

    if constexpr (has_derivative) {
      for (auto &c : stopping_conditions_with_derivative) {
        c->step();
      }
    }
  }

  O current_objective;

  // todo std::variant instead of 2 vectors?
  std::vector<std::shared_ptr<StoppingCondition<p, false, T>>> stopping_conditions;
  std::vector<std::shared_ptr<StoppingCondition<p, true, T>>> stopping_conditions_with_derivative;

  const std::function<bool()> step;
};
}  // namespace opt

#endif
