#pragma once

#include <concepts>
#include <tuple>
#include <type_traits>

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
class network final {
 public:
  // Member Type
  using layers_type =
      std::invoke_result_t<decltype(connect_layers<InputLayer, Layers...>),
                           InputLayer, Layers...>;
  using value_type = decltype(std::apply(
      [](auto&&... layers) { return std::tuple{layers.value()...}; },
      layers_type{}));

  // Constructor
  network() = default;

  [[nodiscard]] explicit constexpr network(const InputLayer&& input_layer,
                                           Layers&&... layers)
      : layers_{connect_layers(input_layer, std::forward<Layers>(layers)...)} {}

  // Getter/Setter
  [[nodiscard]] constexpr auto value() const noexcept -> value_type {
    return std::apply(
        [](auto&&... layers) { return std::tuple{layers.value()...}; },
        layers_);
  }

 private:
  // Member variable
  layers_type layers_{};
};

}  // namespace nou
