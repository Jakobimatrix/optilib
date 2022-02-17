#ifndef EIGEN_UTILS_HPP
#define EIGEN_UTILS_HPP

#include <Eigen/Dense>
#include <array>

#include "definitions.hpp"
#include "math.hpp"
#include "rand.hpp"

namespace opt {

template <int m, int n, class T>
inline Eigen::Matrix<double, n, m> pInv(const Eigen::Matrix<T, m, n>& A) {
  using U = Eigen::Matrix<T, m, m>;
  using S_inv = Eigen::Matrix<T, n, m>;
  using V = Eigen::Matrix<T, n, n>;

  Eigen::JacobiSVD<Eigen::Matrix<T, m, n>> _svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);

  U u = U::Zero();
  S_inv s_inv = S_inv::Zero();
  V v = V::Zero();

  u << _svd.matrixU();
  v << _svd.matrixV();
  s_inv.diagonal() << _svd.singularValues().cwiseInverse();

  return v * s_inv * u.transpose();
}

template <int m, int n, class T>
inline Eigen::Matrix<T, m, m - n> getNullspace(Eigen::Matrix<T, m, n>& A) {
  static_assert(m > n,
                "The given Matrix must have more rows (m) than collumns (n)");
  constexpr int nullspace_dim = m - n;
  // https://stackoverflow.com/questions/34662940/how-to-compute-basis-of-nullspace-with-eigen-library
  // todo see if four fundamental subspaces and URV decomposition is faster.
  Eigen::FullPivLU<Eigen::Matrix<T, n, m>> lu(A.transpose());

  return lu.kernel();
}

template <unsigned n, class T = double>
using XdPoint = Eigen::Matrix<T, n, 1>;


template <unsigned n, class T = double>
T squaredNorm(const XdPoint<n, T>& p) {
  T normsquared = 0;
  for (unsigned i = 0; i < n; ++i) {
    normsquared += p(i, 0) * p(i, 0);
  }
  return normsquared;
}

// Struct to store information about a n-D line between two points p1 and p2.
template <unsigned n, class T>
struct HyperLine {
  using Point = XdPoint<n, T>;

  // line = (p2 - p1)*s + p1
  Point p1;
  Point p2;

  /*!
   * \brief Calculates a point on (p2 - p1)*s + p1 given s.
   * \param s The distance variable from the line.
   * \return The requested point on the line.
   */
  Point getPoint(const double s) const noexcept { return (p2 - p1) * s + p1; }

  /*!
   * \brief Given a point calculate the closest point on this line.
   * \param p The point from which we want to know the closest point on this
   * Line. \return The Point on this Line closest to p.
   */
  Point getClosestPointOnLine(const Point& p) const noexcept {
    /*
    clang-format off
    Closest point pc to p on l(x) = (p2 - p1)*x + p1
    And v1 = (p2 - p1)
    And v2 = (pL - p)
    Than v1 * v2 = 0 (orthogonal)
    pc is somewere on l1 -> pc = l(x) = (p2 - p1)*x + p1

    0 = Summ[ v1_i * v2_i]
      = Summ[ v1_i * (pc - p)_i]
      = Summ[ v1_i * (l(x) - p)_i]
      = Summ[ v1_i * ((p2 - p1)*x + p1 - p)_i]
      = Summ[ v1_i * (v1*x + p1 - p)_i]
      = Summ[ x * v1_i^2 + v1_i^2 * (p1 - p)_i]
    Summ[ x * v1_i^2 ] = Summ[ v1_i^2 * (p1 - p)_i ]
    x * norm(v1)^2 = transposed(v1)*(p1 - p)
    x = transposed(v1)*(p1 - p)/norm(v1)^2
    clang-format on
    */

    const auto v1 = p2 - p1;
    const auto x = v1.dot(p1 - p) / squaredNorm<n, T>(v1);
    return getPoint(-x);  // TODO where does the -1 come from?
  }
};

template <unsigned dim, class T>
struct HyperPlane {
  static constexpr unsigned END_V = dim - 1;
  static constexpr unsigned INDEX_Vn = dim;

  using Point = XdPoint<dim, T>;
  using HyperSpacePoint = XdPoint<dim - 1, T>;
  // X = (P0 - Pn)*t0 + (P1 - Pn)*t1 + ... + (Pn-1 - Pn)*tn-1 + Pn

  // X = Pn + V0*t0 + V1*t1 + ... + Vn-1*tn-1
  std::array<Point, dim> hyper_plane_parameters;

  // for given X -> what is t?
  using XtoT = Eigen::Matrix<T, dim - 1, dim>;
  using TtoX = Eigen::Matrix<T, dim, dim - 1>;
  std::shared_ptr<XtoT> inv_matrix = nullptr;

  HyperPlane() = default;
  HyperPlane(const HyperPlane&) = default;
  // operator=(const HyperPlane&) = default;
  HyperPlane(HyperPlane&&) = default;
  // operator=(HyperPlane&&) = default;

  template <template <class, class> class TContainer, bool GIVEN_POINTS>
  HyperPlane(const TContainer<Point*, std::allocator<Point*>>& container) {
    assert(container.size() == dim &&
           "Given STL container does not contain the number (p) of expected "
           "points.");
    if constexpr (GIVEN_POINTS) {
      for (size_t i = 0; i < END_V; ++i) {
        hyper_plane_parameters[i] = container[i] - container[INDEX_Vn];
      }
      hyper_plane_parameters[INDEX_Vn] = container[INDEX_Vn];
    } else {
      std::copy(container.begin(), container.end(), hyper_plane_parameters);
    }
  }

  /*!
   * \brief Construct the Plane equation from the Form A*X = b
   * \param A Vector of size [n+1 x 1]
   * \param b constant
   */
  /*
  HyperPlane(const Eigen::Matrix<T, 1, dim>& A, T b) {
    Point x_rotated;
    for (unsigned i = 0; i < dim; ++i) {
      x_rotated(i, 0) = static_cast<T>(dim - i);
    }

    const auto rotate = [&x_rotated](unsigned pos) {
      x_rotated.array() = x_rotated.array() + 1;
      x_rotated(pos, 0) = 1;
    };

    unsigned current_pos = 0;
    const T max_a = A.matrix().cwiseAbs().maxCoeff();
    const T epsilon = getEpsilon(max_a);

    int variable_use_count = 0;
    unsigned x_i = 0;
    size_t index = 0;


    for (unsigned i = 0; i < dim; ++i) {
      ++variable_use_count;
      bool zero_round = isNearlyZero(A(0, i), epsilon);

      // In case of leading zeros in A
      while (zero_round && x_i == 0u) {
        ++i;
        ++variable_use_count;
        zero_round = isNearlyZero(A(0, i), epsilon);
      }

      if (!zero_round) {
        x_i = i;
      }

      while (variable_use_count > 0) {
        rotate(i);  // Make sure to have vectors with different bases.

        // [a*x](u != i) + a_i*x_i = b
        const T known_left_side = A * x_rotated - A(0, x_i) * x_rotated(x_i, 0);
        const T x_i_value = (b - known_left_side) / A(0, x_i);

        hyper_plane_parameters[index] = x_rotated;
        hyper_plane_parameters[index](0, x_i) = x_i_value;
        index++;
      }
      assert(index == dim && "Something went wrong.");
      const auto pn = hyper_plane_parameters[INDEX_Vn];
      for (size_t i = 0; i < END_V; ++i) {
        hyper_plane_parameters[i] -= pn;
      }
      hyper_plane_parameters[INDEX_Vn] = pn;
    }
    */

  HyperPlane(const Eigen::Matrix<T, 1, dim>& A, const T b) {
    // clang-format off
    /* 1. Koordinate form -> Parameter form
     a1*x1 + a2*x2 + .. + an*xn = b (I)

     x1 = t1;
     x2 = t2;
     ...
     xi-1 = ti-1
     (I) xi = (b - a1*x1 - a2*x2 - ... - ai-1*xi-1 - ai+1*xi+1 - ... - an*xn)/ai
     with ai != 0;
     xi+1 = ti
     ...
     xn = ti-1

     |0|   |   1  |     |   0  |           |   0    |       |   0    |          |   0  |
     |.|   |   0  |     |   1  |           |   .    |       |   .    |          |   .  |
     |.|   |   .  |     |   0  |           |   .    |       |   .    |          |   .  |
     |.|   |   .  |     |   .  |           |   0    |       |   .    |          |   .  |
     |0|   |   0  |     |   0  |           |   1    |       |   0    |          |   0  |
     |b| + |-a1/ai|t1 + |-a2/ai|t2 + ... + |-ai-1/ai|ti-1 + |-ai+1/ai|  + ... + |-an/ai|tn-1
     |0|   |   0  |     |   0  |           |   0    |       |    1   |          |   0  |
     |.|   |   .  |     |   .  |           |   .    |       |    0   |          |   .  |
     |0|   |   .  |     |   .  |           |   .    |       |    .   |          |   0  |
     |0|   |   0  |     |   0  |           |   0    |       |    0   |          |   1  |
    */
    // clang-format on

    unsigned current_pos = 0;
    const T max_a = A.matrix().cwiseAbs().maxCoeff();
    const T epsilon = getEpsilon(max_a);

    unsigned index_i = dim - 1;
    for (; index_i >= -1; --index_i) {
      if (!isNearlyZero(A(0, index_i), epsilon)) {
        break;
      }
    }

    assert(index_i > -1 && "HyperPlane:: Given A matrix is zero!");

    for (size_t i = 0; i < index_i; ++i) {
      hyper_plane_parameters[i] = Point::Zero();
      hyper_plane_parameters[i](i, 0) = 1;
      hyper_plane_parameters[i](index_i, 0) = -A(0, i) / A(0, index_i);
    }

    for (size_t i = index_i; i < END_V; ++i) {
      hyper_plane_parameters[i] = Point::Zero();
      hyper_plane_parameters[i](i + 1, 0) = 1;
      hyper_plane_parameters[i](index_i, 0) = -A(0, i) / A(0, index_i);
    }
    hyper_plane_parameters[INDEX_Vn] = Point::Zero();
    hyper_plane_parameters[INDEX_Vn](index_i, 0) = b;
  }

  Point operator()(const HyperSpacePoint& t) const noexcept {
    Point ret = hyper_plane_parameters[INDEX_Vn];
    for (size_t i = 0; i < END_V; ++i) {
      ret += hyper_plane_parameters[i] * t(0, i);
    }
    return ret;
  }

  void getKoordinateForm(Eigen::Matrix<T, 1, dim>& A, T& b) const {
    // Parameter form to Normal form:
    // https://www.mathwizurd.com/linalg/2018/11/15/find-a-normal-vector-to-a-hyperplane
    // clang-format off
    /*
    given n-1 parameter vectors in R^n (as v1,v2...)
        |----v1^T----|
    V = |     ...    | in R^(n-1,n)
        |---vn-1^T---|

            |1 0 ... 0  x1 |
    -> V′ = |0 1 ... 0  x2 |
            |    ...       |
            |0 0 ... 1 xn-1|

    Normal to all vi is: [x1, x2, ..., xn-1, 1]^T
    This is also the nullspace of V
    */
    // clang-format on
    auto V = XtoT::Zero();
    for (unsigned i = 0; i < END_V; ++i) {
      V.row(i) = hyper_plane_parameters[i];
    }

    A = getNullspace(V);
    // A[x-p] = 0 // p is any point on the plane, we simply use hyper_plane_parameters[INDEX_Vn]
    // Normal form to Koordinate form:
    // Ax - Ap = 0 >> Na == b and N == A
    b = A.transposed() * hyper_plane_parameters[INDEX_Vn];
  }

  HyperSpacePoint inv(const Point& p) {
    return getInvMatrix() * (p - hyper_plane_parameters[INDEX_Vn]);
  }

  const XtoT& getInvMatrix() {
    if (inv_matrix == nullptr) {
      inv_matrix = std::make_shared<XtoT>();
      TtoX V = TtoX::Zero();
      for (unsigned i = 0; i < END_V; ++i) {
        V.col(i) = hyper_plane_parameters[i];
      }
      *inv_matrix = pInv<dim, dim - 1, T>(V);
    }
    return *inv_matrix;
  }

  /*!
   * \brief Calculates the closest point on the hyper plane to p.
   * \param p Calc closest point on hyper plane to p.
   * \return The closest point on the hyper plane to p.
   */
  Point getClosestPointOnPlane(const Point& p) const noexcept {
    /*
    plane = (p1 - p0)*t1 + (p2 - p0)*t2 + ... + (pn - p0)*tn + p0
    0. pi = p0
    1. split the plane equation in n line equations: li = (pn - pi)*t + pi
    2. for each li {pi = pc = closest point on li to worst}
    3. pi = closest point on plane to worst
    */
    Point mp = hyper_plane_parameters[INDEX_Vn];
    for (size_t i = 0; i < END_V; ++i) {
      const HyperLine<dim, T> li{
          mp, hyper_plane_parameters[i] + hyper_plane_parameters[INDEX_Vn]};
      mp = li.getClosestPointOnLine(p);
    }
    return mp;
  }
};  // namespace opt

template <unsigned p, unsigned q, class T>
Eigen::Matrix<T, p, q> getUniformRandMatrix(T min, T max) {
  const auto rg = UniformDistribution(min, max);
  return Eigen::Matrix<T, p, q>::Zero().unaryExpr(
      [&](float dummy) { return rg->get(); });
}

template <unsigned p, class T>
ObjectiveType<p, T> getRandObjective(T min, T max) {
  const auto rg = UniformDistribution(min, max);
  return ObjectiveType<p, T>::Zero().unaryExpr(
      [&](float dummy) { return rg->get(); });
}

template <unsigned p, class T>
ObjectiveType<p, T> getRandObjective(const ObjectiveType<p, T>& min,
                                     const ObjectiveType<p, T>& max) {
  auto rg = RandomGenerator::getInstance();
  ObjectiveType<p, T> o;
  for (unsigned i = 0; i < p; ++i) {
    o(i, 0) = rg->uniformDistribution(min(i, 0), max(1.0));
  }
}

}  // namespace opt

#endif  // EIGEN_UTILS_HPP
