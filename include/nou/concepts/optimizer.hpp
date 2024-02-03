#pragma once

#include <execution>

namespace nou {

template <class T>
concept optimizer = requires(T t) {
  typename T::real_type;

  {
    t.add_gradient(std::execution::seq, typename T::real_type{})
  } -> std::same_as<void>;
  { t.apply_gradient() } -> std::same_as<void>;
};

}  // namespace nou
