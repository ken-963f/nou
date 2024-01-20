#include "nou/utility/connect_layers.hpp"

#include <boost/ut.hpp>
#include <concepts>
#include <cstddef>

#include "nou/concepts/complete_layer.hpp"
#include "nou/layer/input_layer.hpp"

namespace mock {

template <std::floating_point RealType, std::size_t InputSize,
          std::size_t OutputSize>
struct complete_layer final {
  using real_type = RealType;
  using size_type = std::size_t;

  static constexpr size_type input_size = InputSize;
  static constexpr size_type output_size = OutputSize;
};

template <std::floating_point RealType, std::size_t InputSize,
          std::size_t OutputSize>
constexpr bool operator==(
    const complete_layer<RealType, InputSize, OutputSize>& /*unused*/,
    const complete_layer<RealType, InputSize, OutputSize>& /*unused*/) {
  return true;
}

template <std::size_t OutputSize>
struct incomplete_layer final {
  using size_type = std::size_t;
  static constexpr size_type output_size = OutputSize;

  template <nou::complete_layer PrevLayer>
  constexpr auto make_complete_layer() noexcept {
    return ::mock::complete_layer<typename PrevLayer::real_type,
                                  PrevLayer::output_size, OutputSize>{};
  }
};

}  // namespace mock

auto main() -> int {
  using namespace boost::ut;

  constexpr std::tuple<float, double, long double> test_value = {};

  "Input->Incomplete"_test = [&]<std::floating_point RealType>() {
    static_assert(nou::connect_layers(nou::input_layer<RealType, 1UZ>{},
                                      mock::incomplete_layer<2UZ>{}) ==
                  std::tuple{mock::complete_layer<RealType, 1UZ, 2UZ>{}});
    static_assert(nou::connect_layers(nou::input_layer<RealType, 2UZ>{},
                                      mock::incomplete_layer<1UZ>{}) ==
                  std::tuple{mock::complete_layer<RealType, 2UZ, 1UZ>{}});
  } | test_value;

  "Input->Complete"_test = [&]<std::floating_point RealType>() {
    static_assert(
        nou::connect_layers(nou::input_layer<RealType, 1UZ>{},
                            mock::complete_layer<RealType, 1UZ, 2UZ>{}) ==
        std::tuple{mock::complete_layer<RealType, 1UZ, 2UZ>{}});
    static_assert(
        nou::connect_layers(nou::input_layer<RealType, 2UZ>{},
                            mock::complete_layer<RealType, 2UZ, 1UZ>{}) ==
        std::tuple{mock::complete_layer<RealType, 2UZ, 1UZ>{}});
  } | test_value;

  "Input->InComplete->Complete"_test = [&]<std::floating_point RealType>() {
    static_assert(
        nou::connect_layers(nou::input_layer<RealType, 1UZ>{},
                            mock::incomplete_layer<2UZ>{},
                            mock::complete_layer<RealType, 2UZ, 3UZ>{}) ==
        std::tuple{mock::complete_layer<RealType, 1UZ, 2UZ>{},
                   mock::complete_layer<RealType, 2UZ, 3UZ>{}});
  } | test_value;

  "Input->Complete->InComplete"_test = [&]<std::floating_point RealType>() {
    static_assert(
        nou::connect_layers(nou::input_layer<RealType, 1UZ>{},
                            mock::complete_layer<RealType, 1UZ, 2UZ>{},
                            mock::incomplete_layer<3UZ>{}) ==
        std::tuple{mock::complete_layer<RealType, 1UZ, 2UZ>{},
                   mock::complete_layer<RealType, 2UZ, 3UZ>{}});
  } | test_value;
}
