#pragma once

#include <cstddef>

namespace nou {

template <std::size_t Size>
struct size final {
  static constexpr std::size_t value = Size;
};

}  // namespace nou
