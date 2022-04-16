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

  Objective getExpansion() const noexcept {
    Objective expansion = Objective::Zero();
    auto it = vertices.begin();
    const auto end = vertices.end();
    const auto& best = getBestVertex();
    while (++it != end) {
      const Objective diff_abs = (it->second - best).cwiseAbs();
      expansion = expansion.cwiseMax(diff_abs);
    }
    return expansion;
  }

  std::multimap<T, Objective> vertices;
};

template <unsigned p, typename T = double, bool debug = false>
class SimplexDownhill : public Optimizer<p, false, HESSIAN::NO, T, debug> {
  friend SimplexDownhill<p + 1, T, debug>;

 public:
  using Objective = ObjectiveType<p, T>;
  using ObjectiveFunction = ObjectiveFunctionType<p, T>;

  /*!
   * \brief SimplexDownhill Optimization.
   * \param objective_function The objective function to be minimized.
   * \param initial_guess The best guess of the location of the minima of the objective function.
   * \param initial_step_size A hint about how big the initial simplex needs to be in every search direction.
   */
  SimplexDownhill(const ObjectiveFunction& objective_function,
                  const Objective& initial_guess,
                  const Objective& initial_step_size)
      : Optimizer<p, false, HESSIAN::NO, T, debug>(
            [this]() { return step(); }, objective_function, initial_guess) {
    reset(initial_step_size);
  }

  /*!
   * \brief SimplexDownhill Optimization.
   * \param objective_function The objective function to be minimized.
   * \param initial_guess The best guess of the location of the minima of the objective function.
   */
  SimplexDownhill(const ObjectiveFunction& objective_function, const Objective& initial_guess)
      : Optimizer<p, false, HESSIAN::NO, T, debug>(
            [this]() { return step(); }, objective_function, initial_guess) {
    const Objective initial_step_size = Objective::Ones();
    reset(initial_step_size);
  }

  /*!
   * \brief SimplexDownhill Optimization.
   * \param objective_function The objective function to be minimized.
   * \param initial_guesses
   */
  SimplexDownhill(const ObjectiveFunction& objective_function,
                  const std::multimap<T, Objective>& vertices)
      : Optimizer<p, false, HESSIAN::NO, T, debug>([this]() { return step(); },
                                                   objective_function,
                                                   vertices.begin()->second,
                                                   vertices.begin()->first) {
    simplex.vertices = vertices;
    assert(vertices.size() == p + 1);
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

    // REFLECT
    auto reflection = move_along_line.getPoint(REFELEXION_VALUE);
    const auto direction = reflection - worst_vertex;
    const auto c = this->isObjectiveWithinConstrains(reflection, direction);
    if (c) {
      if ((mirror_point - reflection).norm() < 0.00001 ||
          (reflection - worst_vertex).norm() < 0.00001) {
        return rankLoss(c);
      }
    }
    const T reflection_score = this->J(reflection);

    if (reflection_score < this->getCurrentScore() && c == nullptr) {
      this->debugCurrentStep(reflection, reflection_score);
      // TRY EXPAND
      auto expanded = move_along_line.getPoint(EXPANSION_VALUE);
      this->isObjectiveWithinConstrains(expanded, direction);
      const T expanded_score = this->J(expanded);
      if (expanded_score < reflection_score) {
        // USE EXPANSION
        this->setNewOptimum(expanded_score, expanded);
        simplex.putNewObjective(expanded, expanded_score);
        debugSimplex("expansion");
      } else {
        // USE REFLECTION
        this->setNewOptimum(reflection_score, reflection);
        simplex.putNewObjective(reflection, reflection_score);
        debugSimplex("reflection");
      }
    } else if (reflection_score < simplex.getSecondWorstVertexScore()) {
      // USE REFLECTION
      this->setNewOptimum(reflection_score, reflection);
      simplex.putNewObjective(reflection, reflection_score);
      debugSimplex("reflection");
    } else if (reflection_score < simplex.getWorstVertexScore()) {
      this->debugCurrentStep(reflection, reflection_score);
      // TRY OUTER CONTRACTION
      auto outer_contraction = move_along_line.getPoint(OUTER_CONTRACTION_VALUE);

      if (c) {
        if (!c->isRespected(outer_contraction)) {
          // outer_contraction would be corrected to be equal to reflection
          // and reflection has in this branch the worst score.
          shrink();
          return true;
        }
      }
      const T outer_contraction_score = this->J(outer_contraction);

      if (outer_contraction_score > reflection_score) {
        // USE SHRINK
        shrink();
        return true;
      }
      // USE OUTER_CONTRACTION
      this->setNewOptimum(outer_contraction_score, outer_contraction);
      simplex.putNewObjective(outer_contraction, outer_contraction_score);
      debugSimplex("outer contraction");

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
      debugSimplex("inner contraction");
    }

    return true;
  }

  void shrink() {
    HyperLine<p, T> L;
    L.p2 = simplex.getBestVertex();

    std::multimap<T, Objective> new_verices;
    new_verices.emplace(simplex.getBestVertexScore(), simplex.getBestVertex());

    T index = 0;
    for (auto it = ++simplex.vertices.begin(); it != simplex.vertices.end(); ++it) {
      L.p1 = it->second;
      const auto shrinked = L.getPoint(shrinking_factor(index));
      ++index;
      const auto score = this->J(shrinked);
      new_verices.emplace(score, shrinked);
      this->debugCurrentStep(shrinked, score);
    }
    simplex.vertices.swap(new_verices);
    if (simplex.getBestVertexScore() < this->getCurrentScore()) {
      this->setNewOptimum(simplex.getBestVertexScore(), simplex.getBestVertex());
    }

    debugSimplex("shrink");
  }

  bool rankLoss(std::shared_ptr<Constraint<p, true, T>> constraint) {
    if constexpr (p == 1) {
      return false;  // in 1 d we found the optimum if rank loss.
    } else {
      const auto& hp = constraint->getHyperPlane();

      using Objective_ = ObjectiveType<p - 1, T>;
      const ObjectiveFunctionType<p - 1> J_ = [this, &hp](const Objective_& o) {
        return this->J((*hp)(o));
      };


      std::multimap<T, Objective_> vertices_;
      auto it = simplex.vertices.begin();
      const auto end = --simplex.vertices.end();
      while (it != end) {
        vertices_.emplace(it->first, hp->inv(it->second));
        ++it;
      }

      auto optimizer = std::make_shared<SimplexDownhill<p - 1, T, debug>>(J_, vertices_);

      // const auto initial_guess = hp->inv(simplex.getBestVertex());
      // auto optimizer = std::make_shared<SimplexDownhill<p - 1, T, debug>>(J_, initial_guess);
      return this->searchOnConstraint(constraint, optimizer);
    }
  }

  void reset(const Objective& initial_step_size) {
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

      const auto c = this->isObjectiveWithinConstrains(vertex, direction);
      if (c) {
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

  // |  = intersectionWithHyperPlane
  // WP = worstPoint
  // *  = new point
  static constexpr T REFELEXION_VALUE = 2.;          // .....*...|...WP
  static constexpr T EXPANSION_VALUE = 4.;           // ..*......|...WP
  static constexpr T OUTER_CONTRACTION_VALUE = 3.;   // .......*.|...WP
  static constexpr T INNER_CONTRACTION_VALUE = 0.5;  // .........|.*.WP
  static constexpr T SHRINKING_VALUE_MIN = 0.25;
  static constexpr T SHRINKING_VALUE_MAX = 0.75;
  // Move all vertices in direction of best point.
  // Move the worst vertex SHRINKING_VALUE_MAX and the second_best
  // SHRINKING_VALUE_MIN interpolate for the others
  T shrinking_factor(T index) const noexcept {
    if constexpr (p < 2) {
      return (SHRINKING_VALUE_MAX - SHRINKING_VALUE_MIN) * 0.5;
    } else {
      constexpr T m =
          (SHRINKING_VALUE_MAX - SHRINKING_VALUE_MIN) / (static_cast<T>(p - 1));
      return m * index + SHRINKING_VALUE_MIN;
    }
  }

 public:
  std::vector<std::pair<std::string, Simplex<p, T>>> simplexe;
  void debugSimplex(std::string s) {
    simplexe.push_back(std::make_pair(s, simplex));
  }
};



}  // namespace opt

#endif  // SIMPLEX_DOWNHILL_HPP
