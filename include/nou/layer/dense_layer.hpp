#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <span>
#include <tuple>
#include <type_traits>

#include "nou/concepts/activation_function.hpp"
#include "nou/concepts/complete_layer.hpp"
#include "nou/concepts/initializer.hpp"
#include "nou/concepts/optimizable.hpp"
#include "nou/concepts/optimizer.hpp"
#include "nou/concepts/propagatable.hpp"
#include "nou/core/neuron.hpp"
#include "nou/type_traits/empty_type.hpp"
#include "nou/type_traits/to_span.hpp"

namespace nou {

template <std::size_t Size, class... Ts>
class dense_layer;

template <std::size_t Size, activation_function ActivationFunction,
          initializer Initializer, class... Optimizer>
  requires(sizeof...(Optimizer) <= 1)
class dense_layer<Size, ActivationFunction, Initializer, Optimizer...> final {
 public:
  // Public types
  using size_type = std::size_t;
  using initializer_type = Initializer;

  struct empty_optimizer_type final {
    template <std::floating_point RealType>
    using complete_type = empty_type;
  };

  using optimizer_type =
      std::tuple_element_t<sizeof...(Optimizer),
                           std::tuple<empty_optimizer_type, Optimizer...>>;

  template <complete_layer PrevLayer>
  using complete_layer_type =
      dense_layer<Size,
                  neuron<PrevLayer::output_size, typename PrevLayer::real_type,
                         ActivationFunction,
                         typename optimizer_type::template complete_type<
                             typename PrevLayer::real_type>>>;

  // Public static constants
  static constexpr size_type output_size = Size;

 private:
  // Private member variables
  Initializer initializer_{};
};

template <std::size_t Size, class Node>
  requires propagatable<Node> && optimizable<Node>
class dense_layer<Size, Node> final {
 public:
  // Public types
  using node_type = Node;
  using size_type = std::size_t;
  using real_type = typename node_type::real_type;
  using value_type = std::array<node_type, Size>;

  using input_type = std::span<const real_type, node_type::size>;
  using output_type = std::array<real_type, Size>;
  using backward_type = std::array<real_type, node_type::size>;
  using loss_type = output_type;

  // Public static constants
  static constexpr size_type input_size = node_type::size;
  static constexpr size_type output_size = Size;

  // Constructors

  // Public member functions

 private:
  // Private member variables
  value_type value_{};
};

}  // namespace nou
