#pragma once

#include <concepts>
#include <execution>

namespace nou {

template <class T>
concept optimizable = requires(T t) {
  typename T::output_type;
  {
    t.add_gradient(std::execution::seq, typename T::output_type{},
                   typename T::output_type{})
  } -> std::same_as<void>;
  { t.apply_gradient(std::execution::seq) } -> std::same_as<void>;
};

}
