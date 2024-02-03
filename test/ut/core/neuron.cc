#include "nou/core/neuron.hpp"

#include <boost/ut.hpp>
#include <concepts>
#include <tuple>
#include <type_traits>
#include <utility>

#include "nou/concepts/execution_policy.hpp"
#include "nou/concepts/optimizer.hpp"
#include "nou/type_traits/size.hpp"

namespace mock {

struct activation_function final {
  template <std::floating_point RealType>
  constexpr auto f(RealType x) const noexcept -> RealType {
    return x * RealType{2.0};
  }

  template <std::floating_point RealType>
  constexpr auto df(RealType x) const noexcept -> RealType {
    return x * RealType{-2.0};
  }
};

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

}  // namespace mock

auto main() -> int {
  using namespace boost::ut;

  constexpr std::tuple<float, double, long double> test_value = {};

  "default constructor"_test = []<std::floating_point RealType> {
    static_assert(std::is_nothrow_default_constructible_v<
                  nou::neuron<mock::activation_function>>);
    static_assert(std::is_nothrow_default_constructible_v<nou::neuron<
                      mock::activation_function, mock::incomplete_optimizer>>);

    static_assert(
        std::is_nothrow_default_constructible_v<
            nou::neuron<nou::size<1UZ>, RealType, mock::activation_function>>);
    static_assert(
        std::is_nothrow_default_constructible_v<
            nou::neuron<nou::size<1UZ>, RealType, mock::activation_function,
                        mock::optimizer<RealType>>>);
  } | test_value;
}
