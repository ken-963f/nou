#pragma once

#include <type_traits>

#include "nou/concepts/complete_layer.hpp"
#include "nou/concepts/incomplete_layer.hpp"
#include "nou/concepts/layer.hpp"

namespace nou {

template <class PrevLayer, class Layer>
concept connectable_layers =
    complete_layer<PrevLayer> && complete_layer<Layer> &&
    std::same_as<typename PrevLayer::real_type, typename Layer::real_type> &&
    (PrevLayer::output_size == Layer::input_size);

}  // namespace nou
