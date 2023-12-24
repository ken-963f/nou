#include "nou/network.hpp"

#include <concepts>
#include <type_traits>

#include "boost/ut.hpp"
#include "nou/layer/input_layer.hpp"

namespace mock {

template <std::floating_point RealType, std::size_t OutputSize,
          std::size_t InputSize>
struct complete_layer final {
  using real_type = RealType;
  using size_type = std::size_t;

  static constexpr size_type input_size = InputSize;
  static constexpr size_type output_size = OutputSize;
};

template <std::size_t OutputSize>
struct incomplete_layer final {
  using size_type = std::size_t;
  static constexpr size_type output_size = OutputSize;

  template <nou::complete_layer PrevLayer>
  constexpr auto make_complete_layer() noexcept {
    return complete_layer<typename PrevLayer::real_type, PrevLayer::output_size,
                          OutputSize>{};
  }
};

}  // namespace mock

auto main() -> int {
  using namespace boost::ut;

  constexpr std::tuple<float, double, long double> test_value = {};

  "default constructor"_test = []<std::floating_point RealType>() {
    static_assert(
        std::is_nothrow_default_constructible_v<nou::network<
            nou::input_layer<RealType, 1UZ>, mock::incomplete_layer<2UZ>>>);
    static_assert(
        std::is_nothrow_default_constructible_v<nou::network<
            nou::input_layer<RealType, 1UZ>, mock::incomplete_layer<2UZ>,
            mock::incomplete_layer<3UZ>>>);
  } | test_value;
}
