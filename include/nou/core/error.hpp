#pragma once

#include <cstddef>
#include <string_view>

namespace nou {

struct error final {
  std::string_view what{};
  std::size_t layer_index{};
};

}  // namespace nou
