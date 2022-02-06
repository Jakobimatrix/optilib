#ifndef SIMPLEX_DOWNHILL_HPP
#define SIMPLEX_DOWNHILL_HPP

#include <fstream>
#include <map>

#include "eigen_utils.hpp"
#include "optimizer.hpp"

/*
The Algorithm: Simplex Downhill or Nelder-Mead uses a Simplex (Shape of N+1 Vertices) to
approximate the gradient of a N-D objective Function. It moves the worst (Scored) vertex
in the direction of the approximated gradient: https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method

Advantage:
It will approximate the optimum in few steps.

Disadvantage:
1. Finding the true optimum might take very long depending on the surface. I suggest
to use it only to approximate the optimum and than use Sectioning if a better score is needed.
*/

namespace opt {

template <unsigned p, typename T = double>
struct Simplex {
  using Objective = ObjectiveType<p, T>;

  const Objective& getBestVertex() const noexcept {
    return vertices.begin()->second;
  }
  T getBestVertexScore() const noexcept { return (vertices.begin())->first; }
  const Objective& getWorstVertex() const noexcept {
    return (--vertices.end())->second;
  }
  T getWorstVertexScore() const noexcept { return (--vertices.end())->first; }

  T getSecondWorstVertexScore() const noexcept {
    return (----vertices.end())->first;
  }

  void putNewObjective(const Objective& o, const T score) noexcept {
    auto nh = vertices.extract(--vertices.end());
    nh.key() = score;
    const auto it = vertices.insert(std::move(nh));
    it->second = o;
  }

  Objective getMirrorPoint() const noexcept {
    auto it = vertices.begin();
    const auto end = --vertices.end();
    const auto& worst = getWorstVertex();
    Objective mp = it->second;
    /*
    plane = (p1 - p0)*x1 + (p2 - p0)*x2 + ... + (pn - p0)*xn + p0
    0. pi = p0
    1. split the plane equation in n line equations: li = (pn - pi)*x + pi
    2. for each li {pi = pc = closest point on li to worst}
    3. pi = closest point on plane to worst
    */
    while (++it != end) {
      const HyperLine<p, T> li{mp, it->second};
      mp = li.getClosestPointOnLine(worst);
    }
    return mp;
  }

  std::multimap<T, Objective> vertices;
};

template <unsigned p, typename T = double, bool debug = false>
class SimplexDownhill : public Optimizer<p, false, HESSIAN::NO, T, debug> {
 public:
  using Objective = ObjectiveType<p, T>;
  using ObjectiveFunction = ObjectiveFunctionType<p, T>;

  /*!
   * \brief SimplexDownhill Optimization.
   * \param objective_function The objective function to be minimized.
   * \param initial_guess The best guess of the location of the minima of the objective function.
   * \param starting_stepsize A hint about how big the initial simplex needs to be in every search direction.
   */
  SimplexDownhill(const ObjectiveFunction& objective_function,
                  const Objective& initial_guess,
                  const Objective& step_size)
      : Optimizer<p, false, HESSIAN::NO, T, debug>(
            [this]() { return step(); }, objective_function, initial_guess),
        initial_step_size(step_size) {
    reset();
  }

 private:
  /*!
   * \brief The Simplex Downhill algorithm.
   */
  bool step() noexcept {
    // 1. Get closest point on Hyperplane (spanned by all vertices except worst) to the worst.

    const auto mirror_point = simplex.getMirrorPoint();
    const auto& worst_vertex = simplex.getWorstVertex();
    const HyperLine<p, T> move_along_line{worst_vertex, mirror_point};

    const auto move_along_line_with_constraints =
        [&worst_vertex, &move_along_line, this](T value) {
          Objective point = move_along_line.getPoint(value);
          const auto direction = point - worst_vertex;
          this->isObjectiveWithinConstrains(point, direction);
          return point;
        };

    // REFLECT
    const auto reflection = move_along_line_with_constraints(REFELEXION_VALUE);
    const T reflection_score = this->J(reflection);

    if (reflection_score < this->getCurrentScore()) {
      this->debugCurrentStep(reflection, reflection_score);
      // TRY EXPAND
      const auto expanded = move_along_line_with_constraints(EXPANSION_VALUE);
      const T expanded_score = this->J(expanded);
      if (expanded_score < reflection_score) {
        // USE EXPANSION
        this->setNewOptimum(expanded_score, expanded);
        simplex.putNewObjective(expanded, expanded_score);
      } else {
        // USE REFLECTION
        this->setNewOptimum(reflection_score, reflection);
        simplex.putNewObjective(reflection, reflection_score);
      }
    } else if (reflection_score < simplex.getSecondWorstVertexScore()) {
      // USE REFLECTION
      this->setNewOptimum(reflection_score, reflection);
      simplex.putNewObjective(reflection, reflection_score);
    } else if (reflection_score < simplex.getWorstVertexScore()) {
      this->debugCurrentStep(reflection, reflection_score);
      // TRY OUTER CONTRACTION
      const auto outer_contraction =
          move_along_line_with_constraints(OUTER_CONTRACTION_VALUE);
      const T outer_contraction_score = this->J(outer_contraction);

      if (outer_contraction_score > reflection_score) {
        // USE SHRINK
        shrink();
        return true;
      }
      // USE OUTER_CONTRACTION
      this->setNewOptimum(outer_contraction_score, outer_contraction);
      simplex.putNewObjective(outer_contraction, outer_contraction_score);

    } else {
      this->debugCurrentStep(reflection, reflection_score);
      // TRY INNER CONTRACTION (Can not violate linear constraint)
      const auto inner_contraction = move_along_line.getPoint(INNER_CONTRACTION_VALUE);
      const T inner_contraction_score = this->J(inner_contraction);
      if (inner_contraction_score > simplex.getWorstVertexScore()) {
        // USE SHRINK
        shrink();
        return true;
      }
      // USE INNER_CONTRACTION
      this->setNewOptimum(inner_contraction_score, inner_contraction);
      simplex.putNewObjective(inner_contraction, inner_contraction_score);
    }

    return true;
  }

  void shrink() {
    HyperLine<p, T> L;
    L.p2 = simplex.getBestVertex();

    std::multimap<T, Objective> new_verices;
    new_verices.emplace(simplex.getBestVertexScore(), simplex.getBestVertex());

    for (auto it = ++simplex.vertices.begin(); it != simplex.vertices.end(); ++it) {
      L.p1 = it->second;
      const auto shrinked = L.getPoint(SHRINKING_VALUE);
      const auto score = this->J(shrinked);
      new_verices.emplace(score, shrinked);
      this->debugCurrentStep(shrinked, score);
    }
    simplex.vertices.swap(new_verices);
    if (simplex.getBestVertexScore() < this->getCurrentScore()) {
      this->setNewOptimum(simplex.getBestVertexScore(), simplex.getBestVertex());
    }
  }

  void reset() {
    const auto& best_guess = this->getCurrentOptimum();
    simplex.vertices.clear();
    simplex.vertices.emplace(this->getCurrentScore(), best_guess);

    const auto saveVertex = [this](const Objective& o) {
      const T score = this->J(o);
      simplex.vertices.emplace(score, o);
      this->debugCurrentStep(o, score);
    };

    Objective vertex = best_guess;
    Objective direction = Objective::Zero();
    for (unsigned i = 0; i < p; ++i) {
      direction(i, 0) = initial_step_size(i, 0);
      vertex(i, 0) = best_guess(i, 0) + initial_step_size(i, 0);

      if (!this->isObjectiveWithinConstrains(vertex, direction)) {
        const auto corrected_vertex = vertex;
        direction(i, 0) = -initial_step_size(i, 0);
        vertex(i, 0) = best_guess(i, 0) - initial_step_size(i, 0);

        if (!this->isObjectiveWithinConstrains(vertex, direction)) {
          const auto& new_corrected_vertex = vertex;
          if ((best_guess - corrected_vertex).norm() >
              (best_guess - new_corrected_vertex).norm()) {
            saveVertex(corrected_vertex);
          } else {
            saveVertex(new_corrected_vertex);
          }
        }
      } else {
        saveVertex(vertex);
      }

      vertex(i, 0) = best_guess(i, 0);
      direction(i, 0) = 0;
    }
  }

  Simplex<p, T> simplex;
  const Objective initial_step_size;

  // |  = intersectionWithHyperPlane
  // WP = worstPoint
  // *  = new point
  static constexpr T REFELEXION_VALUE = 2.;          // .....*...|...WP
  static constexpr T EXPANSION_VALUE = 4.;           // ..*......|...WP
  static constexpr T OUTER_CONTRACTION_VALUE = 3.;   // .......*.|...WP
  static constexpr T INNER_CONTRACTION_VALUE = 0.5;  // .........|.*.WP
  static constexpr T SHRINKING_VALUE = 0.5;
  // Move all vertices in direction of best point.
};

}  // namespace opt

#endif  // SIMPLEX_DOWNHILL_HPP
