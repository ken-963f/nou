#pragma once

#include <concepts>
#include <cstddef>
#include <expected>
#include <tuple>
#include <type_traits>
#include <utility>

#include "nou/concepts/activation_function.hpp"
#include "nou/concepts/execution_policy.hpp"
#include "nou/concepts/optimizer.hpp"
#include "nou/core/error.hpp"
#include "nou/type_traits/empty_type.hpp"
#include "nou/type_traits/to_span.hpp"

namespace nou {

template <std::size_t Size, std::floating_point RealType,
          activation_function ActivationFunction, class Optimizer = empty_type>
  requires optimizer<Optimizer> || std::same_as<Optimizer, empty_type>
class neuron final {
 public:
  // Public types
  using size_type = std::size_t;
  using real_type = RealType;

  using input_type = std::array<real_type, Size>;
  using output_type = real_type;
  using backward_type = input_type;
  using loss_type = output_type;

  using activation_function_type = ActivationFunction;

  using weight_type = std::array<real_type, Size>;
  using bias_type = real_type;
  using value_type = std::pair<weight_type, bias_type>;

  using optimizer_type = Optimizer;
  using optimizers_type = std::conditional_t<
      std::same_as<optimizer_type, empty_type>, empty_type,
      std::pair<std::array<optimizer_type, Size>, optimizer_type>>;

  // Public static constants
  static constexpr size_type size = Size;

  // Constructors
  neuron() noexcept = default;

  // Public member functions
  constexpr auto forward_propagate(const execution_policy auto& policy,
                                   to_const_span_t<input_type> input) const
      -> std::expected<output_type, error> {
    // TODO: Implement
    return output_type{};
  }

  constexpr auto backward_propagate(const execution_policy auto& policy,
                                    output_type output, loss_type loss) const
      -> std::expected<backward_type, error> {
    // TODO: Implement
    return backward_type{};
  }

  constexpr void add_gradient(const execution_policy auto& policy,
                              output_type output, loss_type loss) {
    // TODO: Implement
  }

  constexpr void apply_gradient(const execution_policy auto& policy) {}

 private:
  // Private member variables
  value_type value_{};
  [[no_unique_address]] optimizers_type optimizers_{};
};

}  // namespace nou
