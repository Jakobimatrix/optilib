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

/*!
 * \brief Computes the nullspace of Matrix A' (given A)
 * clang-format off
    |----v1^T----|
A = |     ...    | in R^(n,m)
    |---vn-1^T---|

 - Expext A' to have rank n!
 - Expect m > n!
 ->Than the Nullspace (of A') which is orthogonal to all v1 is N in R^(m, m-n)
clang-format on
 *
 * \return Nullspace N in R^(m, m-n)
 */
template <int m, int n, class T>
inline Eigen::Matrix<T, m, m - n> getNullspace(Eigen::Matrix<T, n, m>& A) {
  static_assert(m > n,
                "The given Matrix must have more rows (m) than collumns (n)");
  constexpr int nullspace_dim = m - n;
  // https://stackoverflow.com/questions/34662940/how-to-compute-basis-of-nullspace-with-eigen-library
  // todo see if four fundamental subspaces and URV decomposition is faster.
  Eigen::FullPivLU<Eigen::Matrix<T, n, m>> lu(A);

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

// Struct to store a hyper plane in parameter form.
template <unsigned dim, class T>
struct HyperPlane {
  static constexpr unsigned END_V = dim - 1;
  static constexpr unsigned INDEX_Vn = dim - 1;

  using Point = XdPoint<dim, T>;
  using HyperSpacePoint = XdPoint<dim - 1, T>;
  // X = (P0 - Pn)*t0 + (P1 - Pn)*t1 + ... + (Pn-1 - Pn)*tn-1 + Pn

  // X = Pn + V0*t0 + V1*t1 + ... + Vn-1*tn-1
  std::array<Point, dim> hyper_plane_parameters;  // Last index/point is constant offset of parameter form

  // for given X -> what is t?
  using XtoT = Eigen::Matrix<T, dim - 1, dim>;
  using TtoX = Eigen::Matrix<T, dim, dim - 1>;
  std::shared_ptr<Eigen::ColPivHouseholderQR<TtoX>> x_to_t = nullptr;

  HyperPlane() = default;
  HyperPlane(const HyperPlane&) = default;
  // operator=(const HyperPlane&) = default;
  HyperPlane(HyperPlane&&) = default;
  // operator=(HyperPlane&&) = default;

  /*!
   * \brief Constructor for the nD-Hyperplane given n Points inside the plane.
   * \param points N points inside the Hyperplane.
   */
  HyperPlane(std::array<Point, dim>& points) {
    for (size_t i = 0; i < END_V; ++i) {
      hyper_plane_parameters[i] = points[i] - points[INDEX_Vn];
    }
    hyper_plane_parameters[INDEX_Vn] = points[INDEX_Vn];
  }

  /*!
   * \brief Constructor for the nD-Hyperplane given n Points inside the plane.
   * \param points N points inside the Hyperplane.
   */
  HyperPlane(std::vector<Point>& points) {
    assert(points.size() == dim &&
           "Given STL container does not contain the number (p) of expected "
           "points.");
    for (size_t i = 0; i < END_V; ++i) {
      hyper_plane_parameters[i] = points[i] - points[INDEX_Vn];
    }
    hyper_plane_parameters[INDEX_Vn] = points[INDEX_Vn];
  }

  /*!
   * \brief Constructor for the Hyperplane given in Koordinate form. Calculates
   * the normalized Parameter form (which is the form used for storage)
   * \param A The matrix of the given Koordinate form Ax+b=0.
   * \param b The vector of the given Koordinate form Ax+b=0.
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

    unsigned index_i = 0;
    bool found_nonzero = false;
    for (; index_i < dim; ++index_i) {
      if (!isNearlyZero2(A(0, index_i), epsilon)) {
        found_nonzero = true;
        break;
      }
    }

    assert(found_nonzero && "HyperPlane:: Given A matrix is zero!");

    // Construct the normalized vectors which span the hyperplane.
    for (size_t i = 0; i < index_i; ++i) {
      const T hp_index_i = -A(0, i) / A(0, index_i);
      const T dimension_index_i_length = std::sqrt(1 + hp_index_i * hp_index_i);

      hyper_plane_parameters[i] = Point::Zero();
      hyper_plane_parameters[i](i, 0) = 1. / dimension_index_i_length;
      hyper_plane_parameters[i](index_i, 0) = hp_index_i / dimension_index_i_length;
    }

    for (size_t i = index_i; i < END_V; ++i) {
      const T hp_index_i = -A(0, i + 1) / A(0, index_i);
      const T dimension_index_i_length = std::sqrt(1 + hp_index_i * hp_index_i);

      hyper_plane_parameters[i] = Point::Zero();
      hyper_plane_parameters[i](i + 1, 0) = 1. / dimension_index_i_length;
      hyper_plane_parameters[i](index_i, 0) = hp_index_i / dimension_index_i_length;
    }
    hyper_plane_parameters[INDEX_Vn] = Point::Zero();
    hyper_plane_parameters[INDEX_Vn](index_i, 0) = b;
  }

  /*!
   * \brief Construct the Plane equation from the Form Ax = b
   * \param A Vector of size [n+1 x 1]
   * \param b constant
   */
  void getKoordinateForm(Eigen::Matrix<T, 1, dim>& A, T& b) const {
    // Parameter form to Normal form:
    // https://www.mathwizurd.com/linalg/2018/11/15/find-a-normal-vector-to-a-hyperplane
    // clang-format off
    /*
    given n-1 parameter vectors in R^n (as v1,v2...)
         |----v1^T----|
    V' = |     ...    | in R^(n-1,n)
         |---vn-1^T---|

           |1 0 ... 0  x1 |
    -> V = |0 1 ... 0  x2 |
           |    ...       |
           |0 0 ... 1 xn-1|

    Normal to all vi is: [x1, x2, ..., xn-1, 1]^T
    This is also the nullspace of V
    */
    // clang-format on
    XtoT V = XtoT::Zero();
    for (unsigned i = 0; i < END_V; ++i) {
      V.row(i) = hyper_plane_parameters[i];
    }

    A = getNullspace(V);
    // A[x-p] = 0 // p is any point on the plane, we simply use hyper_plane_parameters[INDEX_Vn]
    // Normal form to Koordinate form:
    // Ax - Ap = 0 >> Ap == b and N == A

    b = A * hyper_plane_parameters[INDEX_Vn];
  }


  /*!
   * \brief Given a parameter point on the hyperplane calculate the point in space.
   * \param t Point in n-1-D Space (in parameterspace of hypherplane)
   * \return Point in n-D Space (Point on hyperplane).
   */
  Point operator()(const HyperSpacePoint& t) const noexcept {
    Point ret = hyper_plane_parameters[INDEX_Vn];
    for (size_t i = 0; i < END_V; ++i) {
      ret += hyper_plane_parameters[i] * t(i, 0);
    }
    return ret;
  }

  /*!
   * \brief Normalizes the vectors spanning the hyperplane.
   */
  void normalize() noexcept {
    for (size_t i = 0; i < END_V; ++i) {
      hyper_plane_parameters[i].normalize();
    }
  }

  /*!
   * \brief Given a point in n-D Space on the hyperplane, calculate the corresponding point in the parameterspace of the hyperplane in n-1-D.
   * \param p Point in n-D Space (Point on hyperplane).
   * \return Point in n-1-D Space (in parameterspace of hypherplane).
   */
  HyperSpacePoint inv(const Point& p) {
    if (x_to_t == nullptr) {
      TtoX V;
      for (unsigned i = 0; i < END_V; ++i) {
        V.col(i) = hyper_plane_parameters[i];
      }
      x_to_t = std::make_shared<Eigen::ColPivHouseholderQR<TtoX>>(V);
      assert(x_to_t->rank() == dim - 1 && "Bad rank");
    }

    const auto b = p - hyper_plane_parameters[INDEX_Vn];
    return x_to_t->solve(b);
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
