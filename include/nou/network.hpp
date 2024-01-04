#pragma once

#include <concepts>
#include <cstddef>
#include <expected>
#include <ranges>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>

#include "concepts/metric.hpp"
#include "nou/concepts/execution_policy.hpp"
#include "nou/concepts/layer.hpp"
#include "nou/concepts/loss_function.hpp"
#include "nou/concepts/optimizable.hpp"
#include "nou/core/error.hpp"
#include "nou/layer/input_layer.hpp"
#include "nou/utility/algorithm.hpp"
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

  using output_layer_type = layer_type<std::tuple_size_v<layers_type> - 1>;

  using input_type = std::span<const real_type, InputLayer::input_size>;

  using output_type = typename output_layer_type::output_type;

  using teacher_type =
      std::span<const real_type, output_layer_type::output_size>;

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
      -> std::expected<output_type, error> {
    return predict<0>(policy, std::move(input));
  }

  [[nodiscard]] constexpr auto predict(
      std::span<const real_type, input_size> input) const
      -> std::expected<output_type, error> {
    return predict<0>(std::execution::seq, std::move(input));
  }

  template <metric<output_type> Metric>
  [[nodiscard]] auto evaluate(const execution_policy auto& policy,
                              std::span<const real_type, input_size> input,
                              teacher_type teacher, Metric&& metric) const
      -> std::expected<real_type, error> {
    auto output = predict(policy, std::move(input));
    return std::move(output).transform(
        [metric = std::forward<Metric>(metric),
         teacher = std::move(teacher)]<class T>(T&& output) {
          return metric(std::forward<T>(output), teacher);
        });
  }

  template <metric<output_type> Metric>
  [[nodiscard]] constexpr auto evaluate(
      std::span<const real_type, input_size> input,
      std::span<const real_type, output_size> teacher, Metric&& metric) const
      -> std::expected<real_type, error> {
    auto output = predict(std::move(input));
    return std::move(output).transform(
        [metric = std::forward<Metric>(metric),
         teacher = std::move(teacher)]<class T>(T&& output) {
          return metric(std::forward<T>(output), teacher);
        });
  }

  template <loss_function<real_type> LossFunction, metric<output_type> Metric>
  auto fit(const execution_policy auto& policy, input_type input,
           teacher_type teacher, LossFunction&& loss_function, Metric&& metric)
      -> std::expected<real_type, error> {
    real_type result{};
    auto metric_function =
        [&result, metic = std::forward<Metric>(metric)]<class T, class U>(
            T&& output, U&& teacher) {
          result = metic(std::forward<T>(output), std::forward<U>(teacher));
        };
    auto fit_result =
        fit<0>(policy, std::move(input), std::move(teacher),
               std::forward<LossFunction>(loss_function), metric_function);

    if (fit_result.has_value()) {
      apply_gradient(policy);
    }

    return fit_result.transform([&](auto&&) { return result; });
  }

  template <loss_function<real_type> LossFunction, metric<output_type> Metric>
  constexpr auto fit(input_type input, teacher_type teacher,
                     LossFunction&& loss_function, Metric&& metric)
      -> std::expected<real_type, error> {
    real_type result{};
    auto metric_function =
        [&result, metic = std::forward<Metric>(metric)]<class T, class U>(
            T&& output, U&& teacher) {
          result = metic(std::forward<T>(output), std::forward<U>(teacher));
        };
    auto fit_result =
        fit<0>(std::execution::seq, std::move(input), std::move(teacher),
               std::forward<LossFunction>(loss_function), metric_function);

    if (fit_result.has_value()) {
      apply_gradient(std::execution::seq);
    }

    return fit_result.transform([&](auto&&) { return result; });
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
  // Static Member
  static constexpr size_type first_optimizable_layer_index =
      []<size_type... I>(std::index_sequence<I...>) {
        bool flag = false;
        return ((!flag && optimizable<layer_type<I>> ? (flag = true, I) : 0) +
                ...);
      }(std::make_index_sequence<last_layer_index + 1>{});

  // Member variable
  layers_type layers_{};

  template <size_type I, execution_policy P>
    requires(I <= last_layer_index)
  static constexpr void (*apply_gradient_function_)(
      network<InputLayer, Layers...>&, const P&) =
      [](auto& network, const P& policy) {
        if constexpr (optimizable<layer_type<I>>) {
          network.template layer<I>().apply_gradient(policy);
        }
      };

  // Private Function
  template <size_type I, std::ranges::random_access_range Input>
    requires(I <= last_layer_index)
  constexpr auto predict(const execution_policy auto& policy,
                         Input&& input) const {
    auto result =
        layer<I>().forward_propagate(policy, std::forward<Input>(input));

    if constexpr (I == last_layer_index) {
      return transform_error<I>(result);
    } else {
      return transform_error<I>(std::move(result))
          .and_then([&, this]<std::ranges::random_access_range T>(T&& value) {
            return predict<I + 1>(policy, std::forward<T>(value));
          });
    }
  }

  template <size_type I, loss_function<real_type> LossFunction,
            std::invocable<output_type, teacher_type> Metric>
    requires(I < first_optimizable_layer_index)
  constexpr auto fit(const execution_policy auto& policy,
                     typename layer_type<I>::input_type input,
                     teacher_type teacher, LossFunction&& loss_function,
                     Metric&& metric) const {
    auto output = transform_error<I>(
        layer<I>().forward_propagate(policy, std::move(input)));
    return std::move(output).and_then([&, this]<class T>(T&& output) {
      return fit<I + 1>(policy, std::forward<T>(output), std::move(teacher),
                        std::forward<LossFunction>(loss_function),
                        std::forward<Metric>(metric));
    });
  }

  template <size_type I, loss_function<real_type> LossFunction,
            std::invocable<output_type, teacher_type> Metric>
    requires(I == first_optimizable_layer_index)
  constexpr auto fit(const execution_policy auto& policy,
                     typename layer_type<I>::input_type input,
                     teacher_type teacher, LossFunction&& loss_function,
                     Metric&& metric) {
    auto output = transform_error<I>(
        layer<I>().forward_propagate(policy, std::move(input)));
    auto loss = output.and_then([&, this]<class T>(T& output) {
      return fit<I + 1>(policy, output, std::move(teacher),
                        std::forward<LossFunction>(loss_function),
                        std::forward<Metric>(metric));
    });
    if (loss.has_value()) {
      layer<I>().add_gradient(policy, std::move(output.value()), loss.value());
    }
    return loss;
  }

  template <size_type I, loss_function<real_type> LossFunction,
            std::invocable<output_type, teacher_type> Metric>
    requires(I > first_optimizable_layer_index && I < last_layer_index)
  constexpr auto fit(const execution_policy auto& policy,
                     typename layer_type<I>::input_type input,
                     teacher_type teacher, LossFunction&& loss_function,
                     Metric&& metric) {
    auto output = transform_error<I>(
        layer<I>().forward_propagate(policy, std::move(input)));
    auto loss = output.and_then([&, this]<class T>(T& output) {
      return fit<I + 1>(policy, output, std::move(teacher),
                        std::forward<LossFunction>(loss_function),
                        std::forward<Metric>(metric));
    });
    if (optimizable<layer_type<I>> && loss.has_value()) {
      layer<I>().add_gradient(policy, output.value(), loss.value());
    }
    return std::move(loss).and_then([&, this]<class T>(T&& loss) {
      return transform_error<I>(layer<I>().backward_propagate(
          policy, std::move(output.value()), std::forward<T>(loss)));
    });
  }

  template <size_type I, loss_function<real_type> LossFunction,
            std::invocable<output_type, teacher_type> Metric>
    requires(I == last_layer_index)
  constexpr auto fit(const execution_policy auto& policy,
                     typename layer_type<I>::input_type input,
                     teacher_type teacher, LossFunction&& loss_function,
                     Metric&& metric) {
    auto output = transform_error<I>(
        layer<I>().forward_propagate(policy, std::move(input)));

    if (output.has_value()) {
      std::forward<Metric>(metric)(output.value(), teacher);
    }

    auto loss = output.transform([&, this](auto& output) {
      typename layer_type<I>::output_type loss{};
      transform(policy, output, std::move(teacher), loss.begin(),
                [loss_function = std::forward<LossFunction>(loss_function)](
                    auto output, auto teacher) {
                  return loss_function.df(output, teacher);
                });
      return loss;
    });

    if (optimizable<layer_type<I>> && loss.has_value()) {
      layer<I>().add_gradient(policy, output.value(), loss.value());
    }

    return std::move(loss).and_then([&, this]<class T>(T&& loss) {
      return transform_error<I>(layer<I>().backward_propagate(
          policy, std::move(output.value()), std::forward<T>(loss)));
    });
  }

  template <execution_policy P>
  constexpr void apply_gradient(const P& policy) {
    auto func_array = [this]<size_type... I>(std::index_sequence<I...>) {
      return std::array{apply_gradient_function_<I, P>...};
    }(std::make_index_sequence<last_layer_index + 1>{});
    for_each(policy, func_array, [&, this](auto func) { func(*this, policy); });
  }

  template <size_type I, class T>
    requires(I <= last_layer_index) &&
            std::same_as<error, typename std::remove_cvref_t<T>::error_type>
  constexpr auto transform_error(T&& error) const noexcept {
    return std::forward<T>(error).transform_error([]<class U>(U&& error) {
      error.layer_index = I;
      return std::forward<U>(error);
    });
  }
};

}  // namespace nou
