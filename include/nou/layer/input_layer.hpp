#pragma once

#include <concepts>
#include <cstddef>

namespace nou {

template <std::floating_point RealType, std::size_t InputSize>
  requires(InputSize > 0)
struct input_layer final {
  // Type Definition
  using real_type = RealType;
  using size_type = std::size_t;

  // Static Member
  static constexpr size_type input_size = InputSize;
  static constexpr size_type output_size = input_size;
};

}  // namespace nou
