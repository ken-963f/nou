#include "nou/layer/dense_layer.hpp"

#include <boost/ut.hpp>
#include <concepts>
#include <expected>
#include <tuple>
#include <type_traits>
#include <utility>

#include "nou/concepts/execution_policy.hpp"
#include "nou/concepts/initializer.hpp"
#include "nou/concepts/optimizer.hpp"
#include "nou/type_traits/size.hpp"
#include "nou/type_traits/to_span.hpp"

namespace mock {

template <std::floating_point RealType>
struct optimizer final {
  using real_type = RealType;
  constexpr void add_gradient(const nou::execution_policy auto& _,
                              RealType x) noexcept {
    temp += x;
  }

  constexpr void apply_gradient() noexcept {
    value = std::exchange(temp, RealType{});
  }

  RealType value{};
  RealType temp{};
};

struct incomplete_optimizer final {
  template <std::floating_point RealType>
  using complete_type = optimizer<RealType>;
};

template <std::floating_point RealType>
struct initializer final {
  constexpr auto operator()() const noexcept -> RealType { return RealType{}; }
};

template <std::floating_point RealType>
struct prev_layer final {
  using real_type = RealType;
  using size_type = std::size_t;

  static constexpr size_type input_size = 1UZ;
  static constexpr size_type output_size = 1UZ;
};

template <class... Ts>
struct node;

template <class IncompleteOptimizer>
struct node<IncompleteOptimizer> final {
  template <std::size_t Size, std::floating_point RealType>
  using complete_type =
      node<nou::size<Size>, RealType,
           typename IncompleteOptimizer::template complete_type<RealType>>;
};

template <std::size_t Size, std::floating_point RealType,
          nou::optimizer Optimizer>
struct node<nou::size<Size>, RealType, Optimizer> final {
  // Public types
  using real_type = RealType;
  using size_type = std::size_t;
  using input_type = std::array<real_type, Size>;
  using output_type = real_type;
  using backward_type = input_type;
  using loss_type = real_type;
  using optimizer_type = Optimizer;
  using value_type = real_type;

  // Public static constants
  static constexpr size_type size = Size;

  // Constructors
  node() = default;

  template <nou::initializer Initializer>
  explicit constexpr node(Initializer initializer) noexcept
      : value{initializer()} {}

  // Public Methods
  constexpr auto forward_propagate(const nou::execution_policy auto& _,
                                   nou::to_const_span_t<input_type> input)
      const noexcept -> std::expected<output_type, nou::error> {
    return input[0] * real_type{2.0};
  };

  constexpr auto backward_propagate(const nou::execution_policy auto& _,
                                    output_type output,
                                    loss_type loss) const noexcept
      -> std::expected<backward_type, nou::error> {
    return backward_type{output * loss};
  }

  constexpr void add_gradient(const nou::execution_policy auto& _,
                              output_type output, loss_type loss) noexcept {
    optimizer.add_gradient(_, output * loss);
  }

  constexpr void apply_gradient(const nou::execution_policy auto& _) noexcept {
    optimizer.apply_gradient();
  }

  // Public Members
  optimizer_type optimizer{};
  value_type value{};
};

}  // namespace mock

auto main() -> int {
  using namespace boost::ut;

  constexpr std::tuple<float, double, long double> test_value = {};

  "default constructor"_test = []<std::floating_point RealType> {
    static_assert(std::is_nothrow_default_constructible_v<
                  nou::dense_layer<1UZ, mock::node<mock::incomplete_optimizer>,
                                   mock::initializer<RealType>>>);
    static_assert(
        std::is_nothrow_default_constructible_v<
            typename nou::dense_layer<1UZ,
                                      mock::node<mock::incomplete_optimizer>,
                                      mock::initializer<RealType>>::
                template complete_layer_type<mock::prev_layer<RealType>>>);

    static_assert(
        std::is_nothrow_default_constructible_v<
            nou::dense_layer<1UZ, mock::node<nou::size<1UZ>, RealType,
                                             mock::optimizer<RealType>>>>);
  } | test_value;

  "make_complete_layer"_test = []<std::floating_point RealType> {
    static_assert(
        nou::dense_layer<1UZ, mock::node<mock::incomplete_optimizer>,
                         mock::initializer<RealType>>::
            template make_complete_layer<mock::prev_layer<RealType>>());
  } | test_value;
}
