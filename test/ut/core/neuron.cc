#include "nou/core/neuron.hpp"

#include <boost/ut.hpp>
#include <concepts>
#include <tuple>
#include <type_traits>
#include <utility>

#include "nou/concepts/execution_policy.hpp"
#include "nou/concepts/optimizer.hpp"

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

template <class... Ts>
struct optimizer;

template <>
struct optimizer<> final {
  template <std::floating_point RealType>
  using complete_type = optimizer<RealType>;
};

template <std::floating_point RealType>
struct optimizer<RealType> final {
  using real_type = RealType;

  constexpr void add_gradient(const nou::execution_policy auto& _,
                              RealType x) noexcept {
    temp += x;
  }

  constexpr void apply_gradient() noexcept {
    value = std::exchange(temp, real_type{});
  }

  RealType value{};
  RealType temp{};
};

}  // namespace mock

auto main() -> int {
  using namespace boost::ut;

  constexpr std::tuple<float, double, long double> test_value = {};

  "default constructor"_test = []<std::floating_point RealType> {
    static_assert(std::is_nothrow_default_constructible_v<
                  nou::neuron<1UZ, RealType, mock::activation_function>>);
    static_assert(std::is_nothrow_default_constructible_v<
                  nou::neuron<1UZ, RealType, mock::activation_function,
                              mock::optimizer<RealType>>>);
  } | test_value;
}
