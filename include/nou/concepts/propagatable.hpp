#pragma once

#include <concepts>
#include <execution>
#include <expected>

#include "nou/core/error.hpp"

namespace nou {

template <class PrevLayer, class Layer>
concept propagatable = requires(const PrevLayer& prev_layer,
                                const Layer& layer) {
  typename Layer::input_type;
  typename Layer::output_type;
  typename Layer::backward_type;
  typename Layer::loss_type;

  {
    layer.forward_propagate(std::execution::seq,
                            typename PrevLayer::output_type{})
  } -> std::same_as<std::expected<typename Layer::output_type, error>>;

  {
    layer.backward_propagate(std::execution::seq, typename Layer::output_type{},
                             typename Layer::loss_type{})
  } -> std::same_as<std::expected<typename Layer::backward_type, error>>;
};

}  // namespace nou