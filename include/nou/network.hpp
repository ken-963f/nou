#pragma once

#include <concepts>

#include "nou/concepts/layer.hpp"
#include "nou/layer/input_layer.hpp"
#include "nou/utility/connect_layers.hpp"

namespace nou {

template <complete_layer InputLayer, layer... Layers>
  requires std::same_as<InputLayer, input_layer<typename InputLayer::real_type,
                                                InputLayer::input_size>> &&
           std::invocable<decltype(connect_layers<InputLayer, Layers...>),
                          InputLayer, Layers...> &&
           (sizeof...(Layers) >= 1)
class network final {};

}  // namespace nou
