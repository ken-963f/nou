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
  };
  { t.apply_gradient(std::execution::seq) };
};
}
