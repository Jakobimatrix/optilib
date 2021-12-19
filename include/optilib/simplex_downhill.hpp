#ifndef SIMPLEX_DOWNHILL_HPP
#define SIMPLEX_DOWNHILL_HPP
#include "optimizer.hpp"

namespace opt {
template <unsigned p, typename T = double, bool enable_step_record = false>
class SimplexDownhill : public Optimizer<p, false, T, enable_step_record> {
 public:
 private:
};

}  // namespace opt

#endif  // SIMPLEX_DOWNHILL_HPP
