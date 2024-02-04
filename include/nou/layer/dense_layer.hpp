#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <utility>

#include "nou/concepts/complete_layer.hpp"
#include "nou/concepts/initializer.hpp"
#include "nou/concepts/optimizable.hpp"
#include "nou/concepts/optimizer.hpp"
#include "nou/concepts/propagatable.hpp"

namespace nou {

template <std::size_t Size, class... Ts>
class dense_layer;

template <std::size_t Size, class IncompleteNode, initializer Initializer>
class dense_layer<Size, IncompleteNode, Initializer> final {
 public:
  // Public types
  using size_type = std::size_t;
  using initializer_type = Initializer;

  template <complete_layer PrevLayer>
  using complete_layer_type =
      dense_layer<Size,
                  typename IncompleteNode::template complete_type<
                      PrevLayer::output_size, typename PrevLayer::real_type>>;

  // Static constants
  static constexpr size_type output_size = Size;

  // Constructors
  dense_layer() = default;
  dense_layer(initializer_type initializer) : initializer_{initializer} {}

  // Public member functions
  template <complete_layer PrevLayer>
  constexpr auto make_complete_layer() const noexcept {
    return dense_layer<
        Size, typename IncompleteNode::template complete_type<
                  PrevLayer::output_size, typename PrevLayer::real_type>>{
        initializer_};
  }

 private:
  initializer_type initializer_{};
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

  using input_type = std::array<real_type, node_type::size>;
  using output_type = std::array<real_type, Size>;
  using backward_type = std::array<real_type, node_type::size>;
  using loss_type = output_type;

  // Public static constants
  static constexpr size_type input_size = node_type::size;
  static constexpr size_type output_size = Size;

  // Constructors
  dense_layer() = default;

  explicit constexpr dense_layer(const value_type& value) : value_{value} {}
  explicit constexpr dense_layer(value_type&& value)
      : value_{std::move(value)} {}

  template <initializer Initializer>
  explicit constexpr dense_layer(Initializer initializer)
      : value_{initialize_nodes(initializer)} {}

  // Public member functions
  constexpr auto value() const noexcept -> value_type { return value_; }

 private:
  // Private member variables
  value_type value_{};

  // Private member functions
  template <initializer Initializer>
  constexpr auto intialize_nodes(Initializer initializer) const -> value_type {
    return [=]<std::size_t... Is>(std::index_sequence<Is...>) {
      return value_type{node_type{(Is, initializer)}...};
    }(std::make_index_sequence<Size>{});
  }
};

}  // namespace nou
