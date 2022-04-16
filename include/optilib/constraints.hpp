#ifndef OPTIMIZATION_CONSTRAINTS
#define OPTIMIZATION_CONSTRAINTS

#include <memory>

#include "definitions.hpp"
#include "eigen_utils.hpp"
#include "math.hpp"

namespace opt {

// <RecordSteps>
template <unsigned p, typename T, class Q, typename Enable = void>
class RecordSteps {};

template <unsigned p, typename T, class Q>
class RecordSteps<p, T, Q, std::enable_if_t<std::is_same<Q, TrueType>::value>> {
  using O = ObjectiveType<p, T>;

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

// <ConstraintVariableHolder>
template <unsigned p, typename T, class Q, typename Enable = void>
struct ConstraintVariableHolder {};

template <unsigned p, typename T, class Q>
struct ConstraintVariableHolder<p, T, Q, std::enable_if_t<std::is_same<Q, TrueType>::value>> {

  ConstraintVariableHolder() {}
  ConstraintVariableHolder(const Eigen::Matrix<T, 1, p> &a, T b) : a(a), b(b) {}
  const Eigen::Matrix<T, 1, p> a;
  const T b;
};

template <unsigned p, typename T, class Q>
struct ConstraintVariableHolder<p, T, Q, std::enable_if_t<std::is_same<Q, FalseType>::value>> {
  T penalty_factor;
};
// </ConstraintVariableHolder>


// Common Constraint must retun 0 or a negative value if a given objective does not violate the constraint. A positive value otherwize.
template <unsigned p, typename T = double>
using CommonConstraint = std::function<T(const ObjectiveType<p, T> &)>;


// A constraint is defined as c(O) >= 0
//* There are no equality constraints in the digital world.
//* Eiter use two inequality constraints or transform your
//* objective function and get rid of the dimension(s) that are fixed.
// A Linear constraint is defined as c(O): a*O - b >= 0
// With:
// Objective O in R^{px1}
// Vector/gradient a in R^{1xp}
// intercept b in R
template <unsigned p, bool is_linear, class T = double>
class Constraint : public ConstraintVariableHolder<p, T, EnableType<is_linear>> {
 public:
  static constexpr bool IS_LINEAR = is_linear;

  using Objective = ObjectiveType<p, T>;
  using A = Eigen::Matrix<T, 1, p>;

  Constraint(const Constraint &other) = default;
  Constraint(Constraint &&other) = default;
  Constraint &operator=(const Constraint &other) = default;
  Constraint &operator=(Constraint &&other) = default;

  /*!
   * \brief Define a linear constraint
   * \param a The Vector (gradient) of the linear constraint
   * \param b The displacement of the linear constraint
   */

  Constraint(const A &a, const T b)
      : ConstraintVariableHolder<p, T, TrueType>(a, b) {}

  /*!
   * \brief Define a nonlinear constraint
   * \param constraint_f Function returning <= 0 if given objective does
   * not violate constraint, a positive value otherwize.
   */
  Constraint(const CommonConstraint<p, T> &constraint_f, const T penalty_factor)
      : ConstraintVariableHolder<p, T, FalseType>(penalty_factor),
        constraint_f(constraint_f) {}

  T operator()(const Objective &o) const { return constraint_f(o); }

  /*!
   * \brief Checks if a given objective is inside the constraint.
   * \param o The objective to check.
   * \return True if the given objective is inside the constraint.
   */
  bool isRespected(const Objective &o) const {
    const auto value = constraint_f(o);
    if (value > 0) {
      return false;
    }
    return value <= static_cast<T>(0.);
  }

  /*!
   * \brief Returns the intersection of the linear gradient with a given line O(t) = A*t + B.
   * \param intercept The intercept (B) of the line.
   * \param gradient The gradient (A) of the Line.
   * \param intersection This will hold the calculated intersection.
   * \return True if the gintersection could be calculated (Fails when given two parallel lines).
   */
  template <class Q = EnableType<is_linear>>
  typename std::enable_if<std::is_same<Q, TrueType>::value, bool>::type getIntersection(
      const Objective &intercept, const Objective &gradient, Objective &intersection) const
      noexcept {
    /* The Vector O(t) = A*t + B
     * must satisfy the constraint a*O(t) = b
     * -> a*[A*t + B] = b
     * a*A*t + a*B = b
     * t = (b - a*B)/a*A
     */

    const T aA = this->a * gradient;
    if (isNearlyZero(aA, 7)) {
      return false;
    }

    const auto t = (this->b - this->a * intercept) / aA;
    intersection = gradient * t + intercept;
    return true;
  }

  /*!
   * \brief Calculate the penalty value (for non linear constraint) for a given objective. This is ment to be added to the value of the objective function.
   * \param o The Objective to check if inside constraint.
   * \return Zero if the objective is not violating the constraint. A positive high value otherwize.
   */
  template <class Q = EnableType<is_linear>>
  typename std::enable_if<std::is_same<Q, FalseType>::value, T>::type getQuadraticPenalty(const Objective &o) {
    const auto penalty = constraint_f(o);
    if (penalty > 0.) {
      return this->penalty_factor * penalty * penalty;
    }
    return 0.;
  }


  const std::shared_ptr<HyperPlane<p, T>> getHyperPlane() noexcept {
    if (hyper_plane == nullptr) {
      calculateHyperplaneParameters();
    }
    return hyper_plane;
  }


 private:
  void calculateHyperplaneParameters() {
    hyper_plane = std::make_shared<HyperPlane<p, T>>(this->a, this->b);
  }

  // This is always avaiable
  const CommonConstraint<p, T> constraint_f = [this](const Objective &o) {
    return -this->a * o + this->b;
  };

  std::shared_ptr<HyperPlane<p, T>> hyper_plane;
};

}  // namespace opt
#endif
