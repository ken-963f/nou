#include "nou/network.hpp"

#include <concepts>
#include <type_traits>

#include "boost/ut.hpp"
#include "nou/layer/input_layer.hpp"

namespace mock {

template <std::floating_point RealType, std::size_t InputSize,
          std::size_t OutputSize>
struct complete_layer final {
  // Member Type
  using real_type = RealType;
  using size_type = std::size_t;
  using value_type = real_type;

  // Static Member
  static constexpr size_type input_size = InputSize;
  static constexpr size_type output_size = OutputSize;

  // Constructor
  complete_layer() = default;
  [[nodiscard]] explicit constexpr complete_layer(real_type value)
      : value_{value} {}

  // Getter/Setter
  [[nodiscard]] constexpr auto value() const noexcept -> real_type {
    return value_;
  }

 private:
  real_type value_{};
};

template <std::size_t OutputSize>
struct incomplete_layer final {
  // Member Type
  using size_type = std::size_t;

  // Static Member
  static constexpr size_type output_size = OutputSize;

  // Public Functions
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

  "constructor"_test = []<std::floating_point RealType>() {
    constexpr auto value1 = nou::network{nou::input_layer<RealType, 1UZ>{},
                                         mock::incomplete_layer<2UZ>{},
                                         mock::incomplete_layer<3UZ>{}}
                                .value();
    static_assert(std::get<0>(value1) == RealType{});
    static_assert(std::get<1>(value1) == RealType{});

    constexpr auto value2 =
        nou::network{nou::input_layer<RealType, 1UZ>{},
                     mock::complete_layer<RealType, 1UZ, 2UZ>{1.0},
                     mock::complete_layer<RealType, 2UZ, 3UZ>{2.0}}.value();
    static_assert(std::get<0>(value2) == RealType{1.0});
    static_assert(std::get<1>(value2) == RealType{2.0});
  } | test_value;
}
