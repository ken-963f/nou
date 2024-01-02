#pragma once

#include <concepts>
#include <cstddef>
#include <expected>
#include <ranges>
#include <span>
#include <tuple>
#include <type_traits>

#include "concepts/metric.hpp"
#include "nou/concepts/execution_policy.hpp"
#include "nou/concepts/layer.hpp"
#include "nou/core/error.hpp"
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
  using size_type = std::size_t;
  using real_type = typename InputLayer::real_type;

  using layers_type =
      std::invoke_result_t<decltype(connect_layers<InputLayer, Layers...>),
                           InputLayer, Layers...>;
  using value_type = decltype(std::apply(
      [](auto&&... layers) { return std::tuple{layers.value()...}; },
      layers_type{}));

  template <size_type I>
    requires(I < std::tuple_size_v<layers_type>)
  using layer_type = std::tuple_element_t<I, layers_type>;

  using output_type =
      typename layer_type<std::tuple_size_v<layers_type> - 1>::output_type;

  // Static Member
  static constexpr size_type last_layer_index =
      std::tuple_size_v<layers_type> - 1;
  static constexpr size_type input_size = InputLayer::input_size;
  static constexpr size_type output_size =
      layer_type<last_layer_index>::output_size;

  // Constructor
  network() = default;

  [[nodiscard]] explicit constexpr network(const InputLayer&& input_layer,
                                           Layers&&... layers)
      : layers_{connect_layers(input_layer, std::forward<Layers>(layers)...)} {}

  // Public Functions
  [[nodiscard]] auto predict(const execution_policy auto& policy,
                             std::span<const real_type, input_size> input) const
      -> output_type {
    return predict<0>(policy, std::move(input));
  }

  [[nodiscard]] constexpr auto predict(
      std::span<const real_type, input_size> input) const -> output_type {
    return predict<0>(std::execution::seq, std::move(input));
  }

  template <metric<typename output_type::value_type> Metric>
  [[nodiscard]] auto evaluate(const execution_policy auto& policy,
                              std::span<const real_type, input_size> input,
                              std::span<const real_type, output_size> teacher,
                              Metric&& metric) const
      -> std::expected<real_type, error> {
    output_type output = predict(policy, std::move(input));
    return std::move(output).transform(
        [metric = std::forward<Metric>(metric),
         teacher = std::move(teacher)]<class T>(T&& output) {
          return metric(std::forward<T>(output), teacher);
        });
  }

  template <metric<typename output_type::value_type> Metric>
  [[nodiscard]] constexpr auto evaluate(
      std::span<const real_type, input_size> input,
      std::span<const real_type, output_size> teacher, Metric&& metric) const
      -> std::expected<real_type, error> {
    output_type output = predict(std::move(input));
    return std::move(output).transform(
        [metric = std::forward<Metric>(metric),
         teacher = std::move(teacher)]<class T>(T&& output) {
          return metric(std::forward<T>(output), teacher);
        });
  }

  // Getter/Setter
  [[nodiscard]] constexpr auto value() const noexcept -> value_type {
    return std::apply(
        [](auto&&... layers) { return std::tuple{layers.value()...}; },
        layers_);
  }

  template <size_type I>
    requires(I <= last_layer_index)
  [[nodiscard]] constexpr auto layer() const noexcept -> const layer_type<I>& {
    return std::get<I>(layers_);
  }

  template <size_type I>
    requires(I <= last_layer_index)
  [[nodiscard]] constexpr auto layer() noexcept -> layer_type<I>& {
    return std::get<I>(layers_);
  }

 private:
  // Member variable
  layers_type layers_{};

  // Private Function
  template <size_type I, std::ranges::random_access_range Input>
    requires(I <= last_layer_index)
  constexpr auto predict(const execution_policy auto& policy,
                         Input&& input) const {
    typename layer_type<I>::output_type result =
        layer<I>().forward_propagate(policy, std::forward<Input>(input));

    if constexpr (I == last_layer_index) {
      return result.transform_error([](auto& error) {
        error.layer_index = I;
        return error;
      });
    } else {
      return std::move(result)
          .transform_error([](auto&& error) {
            error.layer_index = I;
            return error;
          })
          .and_then([&, this]<std::ranges::random_access_range T>(T&& value) {
            return predict<I + 1>(policy, std::forward<T>(value));
          });
    }
  }
};

}  // namespace nou
