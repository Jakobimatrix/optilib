#ifndef RAND_HPP
#define RAND_HPP

// taken from https://github.com/Jakobimatrix/randomGenerator/tree/main/include
#include <cassert>
#include <memory>
#include <random>

namespace opt {

class RandomGenerator {

 private:
  /*!
   * \brief private Constructor initializing the generator.
   */
  RandomGenerator() { generator = std::default_random_engine(rd()); }

  // Stop the compiler generating methods of copy the object
  RandomGenerator(RandomGenerator const& copy) = delete;  // Not Implemented
  RandomGenerator& operator=(RandomGenerator const& copy) = delete;  // Not Implemented

 public:
  /*!
   * \brief private Get the one instance ot the class
   * \return A shared pointer to the only existing instance of this class.
   */
  static std::shared_ptr<RandomGenerator> getInstance() {
    static std::shared_ptr<RandomGenerator> instance(new RandomGenerator);

    return instance;
  }

  /*!
   * \brief Get a random number between min and max following a uniform distribution.
   * \param min The lower bound of the uniform distribution
   * \param max The higher bound of the uniform distribution.
   * \return A random value between min and max of type T.
   */
  template <class T>
  T uniformDistribution(T min, T max) {
    static_assert(std::is_arithmetic<T>::value, "The type T must be numeric");
    assert("getRandomGenerator::uniform_distribution: given min > max" && min < max);
    if constexpr (std::is_floating_point<T>::value) {
      std::uniform_real_distribution<> distribution(min, max);
      return distribution(generator);
    } else if constexpr (std::is_integral<T>::value) {
      std::uniform_int_distribution<T> distribution(min, max);
      return distribution(generator);
    }
    assert(
        "RandomGenerator::uniform_distribution: How did you find a type which "
        "IS numeric but neither float nor integral???" &&
        false);
  }

  /*!
   * \brief Get a random number according to a normal distribution.
   * \param mean The mean value of the normal distribution.
   * \param variance The variance of the normal distribution (== the second central moment == standard deviation squared).
   * \return A random value between -inf and inf of type T.
   */
  template <class T = double>
  T normalDistribution(T mean, T variance) {
    static_assert(std::is_floating_point<T>::value,
                  "The type T must be floating point");
    std::normal_distribution<T> distribution(mean, variance);
    return distribution(generator);
  }

  /*!
   * \brief Returns true or false with probablity of 1/2.
   * \return A randomly true or false with probablity of 1/2.
   */
  bool trueFalse() {
    if (zeroOne()) {
      return true;
    }
    return false;
  }

  /*!
   * \brief Returns 0 or 1 with probablity of 1/2.
   * \return A randomly 1 or 0 with probablity of 1/2.
   */
  int zeroOne() {
    std::uniform_int_distribution<int> distribution(0, 1);
    return distribution(generator);
  }

 private:
  std::random_device rd;
  std::default_random_engine generator;
};

/*!
 * \brief Conveniendt Class to hold one specific uniform distribution.
 */
template <class T = double>
class UniformDistribution {
  using T_ = T;

 public:
  /*!
   * \brief UniformDistribution default constructor constructs a uniform distribution between 0. and 1.
   */
  UniformDistribution()
      : UniformDistribution(static_cast<T>(0.), static_cast<T>(1.)) {}

  /*!
   * \brief UniformDistribution constructor constructs a uniform distribution
   * between min and max. \param min The left bound of the uniform distribution.
   * \param max the right bound of the uniform distribution.
   */
  UniformDistribution(T min, T max) : min(min), max(max) {
    rg = rg->getInstance();
  }

  /*!
   * \brief get a random value according to the defined uniform distribution
   * \return a value between min and max.
   */
  T get() const { return rg->uniformDistribution<T>(min, max); }

  std::shared_ptr<RandomGenerator> rg;
  T min;
  T max;
};

/*!
 * \brief Conveniendt Class to hold one specific normal distribution
 */
template <class T = double>
class NormalDistribution {
 public:
  /*!
   * \brief NormalDistribution default constructor constructs a N(0,1) distribution
   */
  NormalDistribution()
      : NormalDistribution(static_cast<T>(0.), static_cast<T>(1.)){};

  /*!
   * \brief NormalDistribution default constructor constructs a N(mean,var) distribution
   * \param mean The first central moment of the distribution
   * \param var The second central moment of the distribution
   */
  NormalDistribution(T mean, T var) : mean(mean), var(var) {
    rg = rg->getInstance();
  };


  /*!
   * \brief get a random value according to the defined normal distribution
   * \return a value between min and max.
   */
  T get() const { return rg->normalDistribution(mean, var); }
  std::shared_ptr<RandomGenerator> rg;
  T mean;
  T var;
};

}  // namespace opt

#endif  // RAND_HPP
