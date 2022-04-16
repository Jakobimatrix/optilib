#ifndef OPTIMIZATION_OPTIMIZE
#define OPTIMIZATION_OPTIMIZE

#include <chrono>
#include <fstream>
#include <memory>
#include <vector>

#include "constraints.hpp"
#include "definitions.hpp"
#include "stoppingcondition.hpp"

namespace opt {

template <unsigned p, class T = double>
struct DebugInfo {
  using Objective = ObjectiveType<p, T>;
  Objective objective;
  T score;
  unsigned long long time;

  static void printHeader(std::ofstream &file, const char SEPERATOR) {
    file << "time (ns)" << SEPERATOR << "score";
    for (int i = 0; i < p; ++i) {
      file << SEPERATOR << "x" << i;
    }
    file << '\n';
  }

  void print(std::ofstream &file, const char SEPERATOR) const {
    file << time << SEPERATOR << score;
    for (unsigned int i = 0; i < p; ++i) {
      file << SEPERATOR << objective(i, 0);
    }
    file << '\n';
  }

  static void print(std::ofstream &open_file,
                    const char SEPERATOR,
                    const std::vector<DebugInfo<p, T>> &info) {
    if (!open_file.is_open() || open_file.bad()) {
      return;
    }
    printHeader(open_file, SEPERATOR);
    for (const auto &i : info) {
      i.print(open_file, SEPERATOR);
    }
  }

  unsigned long long static now() {
    timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return static_cast<unsigned long long>(now.tv_nsec) +
           static_cast<unsigned long long>(now.tv_sec) * 1'000'000'000ul;
  }
};



enum HESSIAN { AVAIABLE_QUADRATIC, AVAIABLE, APPROXIMATE, NO };

template <unsigned p, bool has_derivative, HESSIAN hessian, typename T = double, bool debug = false>
class Optimizer {
  friend Optimizer<p + 1, has_derivative, hessian, T, debug>;

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
    c->init(&current_score, &current_objective, &current_objective_derivative);
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
    c->init(&current_score, &current_objective);
    stopping_conditions.push_back(c);
  }

  void setConstraint(std::shared_ptr<Constraint<p, true, T>> c) {
    linear_constraints.push_back(c);
  }

  void setConstraint(std::shared_ptr<Constraint<p, false, T>> c) {
    nonlinear_constraints.push_back(c);
  }


  /*!
   * \brief Start the optimization until the optimizer
   * found either the optimum or a stoppingcondition applyes.
   */
  void start() noexcept {
    time_start = DebugInfo<p, T>::now();
    if constexpr (debug) {
      debugCurrentStep(current_objective, current_score);
    }

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
  Optimizer(const std::function<bool()> &step_function,
            const ObjectiveFunction &objective_function,
            const Objective &initial_guess)
      : objective_function(objective_function),
        step(step_function),
        current_objective(initial_guess),
        current_score(J(initial_guess)){};

  Optimizer(const std::function<bool()> &step_function,
            const ObjectiveFunction &objective_function,
            const Objective &initial_guess,
            const T initial_guess_score)
      : objective_function(objective_function),
        step(step_function),
        current_objective(initial_guess),
        current_score(initial_guess_score){};

  Optimizer(const std::function<bool()> &step_function,
            const ObjectiveFunction &objective_function,
            const ObjectiveFunctionDerivative &objective_function_derivative,
            const Objective &initial_guess)
      : objective_function(objective_function),
        objective_function_derivative(objective_function_derivative),
        step(step_function),
        current_objective(initial_guess),
        current_objective_derivative(dJ(initial_guess)),
        current_score(J(initial_guess)){};

  template <class Q = EnableType<has_derivative>>
  typename std::enable_if<std::is_same<Q, TrueType>::value, Objective &>::type getCurrentObjectiveDerivative() noexcept {
    return current_objective_derivative;
  }

  template <class Q = EnableType<hessian == AVAIABLE_QUADRATIC>>
  typename std::enable_if<std::is_same<Q, TrueType>::value, std::shared_ptr<Constraint<p, true, T>>>::type
  isObjectiveWithinConstrains(Objective &o) {
    assert(false && "NOT IMPLEMENTED YET");

    // active set method
    return nullptr;
  }

  /*!
   * \brief Checks if a given objective is inside all linear constraints. If it is outside
   * the given objective will be set to the intersection between the constraint and the
   * vector 'from the given objective in direction of griven gradient'
   * \param o Given Objective to correct if necessarry.
   * \param gradient The direction in which to correct the objective if necessarry.
   * \return The violated constraint or a nullpointer if no constraint is violated.
   */
  template <class Q = EnableType<hessian == AVAIABLE_QUADRATIC>>
  typename std::enable_if<std::is_same<Q, FalseType>::value, std::shared_ptr<Constraint<p, true, T>>>::type
  isObjectiveWithinConstrains(Objective &o, const Objective &gradient) {
    std::shared_ptr<Constraint<p, true, T>> violated_constraint = nullptr;
    if (linear_constraints.empty()) {
      return violated_constraint;
    }

    for (const auto &c : linear_constraints) {
      if (!c->isRespected(o)) {
        if (c->getIntersection(o, gradient, o)) {
          violated_constraint = c;
        }
      }
    }
    return violated_constraint;
  }

  template <class Q = EnableType<has_derivative>>
  typename std::enable_if<std::is_same<Q, FalseType>::value, bool>::type setNewOptimum(
      T score, const Objective &o) {
    current_objective = o;
    current_score = score;

    if constexpr (debug) {
      debugCurrentStep(o, score);
    }
    return true;
  }

  template <class Q = EnableType<has_derivative>>
  typename std::enable_if<std::is_same<Q, TrueType>::value, bool>::type setNewOptimum(
      T score, const Objective &o, const Objective &od) {
    current_objective_derivative = od;
    current_objective = o;
    current_score = score;

    if constexpr (debug) {
      debugCurrentStep(o, score);
    }

    return true;
  }

  void resetOptimizer() {
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

 protected:
  bool searchOnConstraint(std::shared_ptr<Constraint<p, true, T>> c,
                          std::shared_ptr<Optimizer<p - 1, has_derivative, hessian, T, debug>> optimizer) {
    const auto hp = c->getHyperPlane();
    optimizer->time_start = time_start;
    // Todo add (active) linear constraints to optimizer.

    const auto step_ = [this, &optimizer, &hp]() {
      const bool res = optimizer->step();
      if constexpr (debug) {
        for (const auto &info : optimizer->debug_info) {
          debug_info.push_back({(*hp)(info.objective), info.score, info.time});
        }
        optimizer->debug_info.clear();
        // neues optimum zuwisen
      }

      return res;
    };

    // start optimization
    do {
      updateStoppingConditions();
    } while (!stop() && step_());
    // todo how to leave constraint? We never know if we found optimum on constraint...

    this->current_objective = (*hp)(optimizer->current_objective);
    this->current_score = optimizer->current_score;
    return false;
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

  const ObjectiveFunction objective_function;
  const ObjectiveFunctionDerivative objective_function_derivative;

  T current_score = std::numeric_limits<T>::max();
  Objective current_objective;
  Objective current_objective_derivative;

  // DEBUGGING
 public:
  template <class Q = EnableType<debug>>
  typename std::enable_if<std::is_same<Q, TrueType>::value, const std::vector<DebugInfo<p, T>> &>::type getDebugInfo() const
      noexcept {
    return debug_info;
  }

 protected:
  template <class Q = EnableType<debug>>
  typename std::enable_if<std::is_same<Q, TrueType>::value, void>::type debugCurrentStep(
      const Objective &o, const T score) noexcept {
    const auto now = DebugInfo<p, T>::now();
    const auto duration = now - time_start;
    debug_info.push_back({o, score, duration});
  }

  template <class Q = EnableType<debug>>
  typename std::enable_if<std::is_same<Q, FalseType>::value, void>::type debugCurrentStep(
      const Objective &o, const T score) noexcept {
    // nop
  }

 private:
  std::vector<DebugInfo<p, T>> debug_info;
  unsigned long long time_start;
};
}  // namespace opt

#endif
