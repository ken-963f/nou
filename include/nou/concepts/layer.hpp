#pragma once

#include "nou/concepts/complete_layer.hpp"
#include "nou/concepts/incomplete_layer.hpp"

namespace nou {

template <class T>
concept layer = complete_layer<T> || incomplete_layer<T>;

}  // namespace nou
