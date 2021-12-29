#ifndef OPTIMIZATION_OPTIMIZE
#define OPTIMIZATION_OPTIMIZE

#include <memory>
#include <vector>

#include "constraints.hpp"
#include "definitions.hpp"
#include "stoppingcondition.hpp"

namespace opt {


template <unsigned p, bool has_derivative, typename T = double>
class Optimizer {
 public:
  using Objective = ObjectiveType<p, T>;
  using ObjectiveFunction = ObjectiveFunctionType<p, T>;
  using ObjectiveFunctionDerivative = ObjectiveFunctionDerivativeType<p, T>;

  const Objective &getCurrentOptimum() const noexcept {
    return current_objective;
  }

  T getCurrentScore() const noexcept { return current_score; }

  template <class Q = EnableType<has_derivative>>
  typename std::enable_if<std::is_same<Q, TrueType>::value, void>::type setStoppingCondition(
      std::shared_ptr<StoppingCondition<p, true, T>> c) {
    if (c == nullptr) {
      return;
    }
    c->init(&current_objective, &current_objective_derivative);
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

  /*!
   * \brief Start the optimization until the optimizer
   * found either the optimum or a stoppingcondition applyes.
   */
  void start() noexcept {
    if (!step()) {
      return;
    }
    do {
      updateStoppingConditions();
    } while (!stop() && step());
  }

  /*!
   * \brief Calculate the score of the given objective function.
   * If non linear constraints where defined, quadratic penalty is used.
   * \param o The objective value at which to evaluate the objective function.
   * \return the score of the objective function.
   */
  T J(const Objective &o) const {
    T score = objective_function(o);
    for (const auto &c : nonlinear_constraints) {
      score += c->getQuadraticPenalty(o);
    }
    return score;
  }

  /*!
   * \brief Calculate the gradient of the objectife function at the given
   * objective
   * \param o The objective value at which to evaluate the objective
   * functions gradient.
   * \return the gradient.
   */
  template <class Q = EnableType<has_derivative>>
  typename std::enable_if<std::is_same<Q, TrueType>::value, Objective>::type dJ(const Objective &o) {
    return this->objective_function_deviation(o);
  }

 protected:
  Optimizer(const std::function<bool()> &step_function, const ObjectiveFunction &objective_function)
      : objective_function(objective_function), step(step_function){};

  Optimizer(const std::function<bool()> &step_function,
            const std::function<const Objective &()> &getCurrentObjective)
      : objective_function(objective_function), step(step_function){};

  template <class Q = EnableType<has_derivative>>
  typename std::enable_if<std::is_same<Q, TrueType>::value, Objective &>::type getCurrentObjectiveDerivative() noexcept {
    return current_objective_derivative;
  }

  template <class Q = EnableType<has_derivative>>
  typename std::enable_if<std::is_same<Q, FalseType>::value, bool>::type setNewOptimum(
      T score, const Objective &o) {
    // todo linear constraints, active set methode
    current_objective = o;
    current_score = score;
  }

  template <class Q = EnableType<has_derivative>>
  typename std::enable_if<std::is_same<Q, TrueType>::value, bool>::type setNewOptimum(
      T score, const Objective &o, const Objective &od) {
    // todo linear constraints, active set methode
    current_objective_derivative = od;
    current_objective = o;
    current_score = score;
  }

  void resetOptimizer() {
    current_score = J(current_objective);
    if constexpr (has_derivative) {
      current_objective_derivative = dJ(current_objective);
    }

    for (auto &c : stopping_conditions) {
      c->reset();
    }
    if constexpr (has_derivative) {
      for (auto &c : stopping_conditions_with_derivative) {
        c->reset();
      }
    }

    // todo constraints
  }

 private:
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

  // todo std::variant instead of 2 vectors?
  std::vector<std::shared_ptr<StoppingCondition<p, false, T>>> stopping_conditions;
  std::vector<std::shared_ptr<StoppingCondition<p, true, T>>> stopping_conditions_with_derivative;

  std::vector<std::shared_ptr<Constraint<p, true, T>>> linear_constraints;
  std::vector<std::shared_ptr<Constraint<p, false, T>>> nonlinear_constraints;

  const std::function<bool()> step;

  T current_score = std::numeric_limits<T>::max();
  Objective current_objective;
  Objective current_objective_derivative;

  const ObjectiveFunction objective_function;
  const ObjectiveFunctionDerivative objective_function_derivative;
};
}  // namespace opt

#endif
