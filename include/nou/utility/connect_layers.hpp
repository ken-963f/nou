#pragma once

#include <tuple>
#include <type_traits>
#include <utility>

#include "nou/concepts/complete_layer.hpp"
#include "nou/concepts/connectable_layers.hpp"
#include "nou/concepts/incomplete_layer.hpp"
#include "nou/concepts/layer.hpp"
#include "nou/layer/input_layer.hpp"

namespace nou {

namespace detail {

template <complete_layer PrevLayer, complete_layer Layer>
  requires connectable_layers<PrevLayer, Layer>
constexpr auto make_complete_layer(Layer&& layer) -> Layer {
  return std::forward<Layer>(layer);
}

template <complete_layer PrevLayer, incomplete_layer Layer>
constexpr auto make_complete_layer(Layer&& layer) -> complete_layer auto {
  return make_complete_layer<PrevLayer>(
      std::forward<Layer>(layer).template make_complete_layer<PrevLayer>());
}

template <class Tuple, layer Layer, layer... Layers>
  requires complete_layer<
      std::tuple_element_t<std::tuple_size_v<std::remove_cvref_t<Tuple>> - 1,
                           std::remove_cvref_t<Tuple>>>
constexpr auto connect_layers(Tuple&& tuple, Layer&& layer,
                              Layers&&... layers) {
  using tuple_t = std::remove_cvref_t<Tuple>;
  using prev_layer_t =
      std::tuple_element_t<std::tuple_size_v<tuple_t> - 1, tuple_t>;
  auto temp = std::apply(
      [layer = std::forward<Layer>(layer)]<class... Ts>(Ts&&... ts) mutable {
        return std::tuple{
            std::forward<Ts>(ts)...,
            make_complete_layer<prev_layer_t>(std::forward<Layer>(layer))};
      },
      std::forward<Tuple>(tuple));

  if constexpr (sizeof...(Layers) == 0) {
    return temp;
  } else {
    return connect_layers(std::move(temp), std::forward<Layers>(layers)...);
  }
}

}  // namespace detail

template <complete_layer InputLayer, layer FirstLayer, layer... Layers>
  requires std::same_as<InputLayer, input_layer<typename InputLayer::real_type,
                                                InputLayer::input_size>>
constexpr auto connect_layers(const InputLayer& /**/, FirstLayer&& first_layer,
                              Layers&&... layers) {
  auto temp = std::tuple{detail::make_complete_layer<InputLayer>(
      std::forward<FirstLayer>(first_layer))};

  if constexpr (sizeof...(Layers) == 0) {
    return temp;
  } else {
    return detail::connect_layers(std::move(temp),
                                  std::forward<Layers>(layers)...);
  }
}

}  // namespace nou
