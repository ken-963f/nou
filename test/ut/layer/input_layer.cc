#include "nou/layer/input_layer.hpp"

#include <concepts>
#include <limits>
#include <tuple>
#include <utility>

#include "boost/ut.hpp"
#include "nou/concepts/complete_layer.hpp"

auto main() -> int {
  using namespace boost::ut;

  constexpr std::tuple<float, double, long double> test_value = {};
  constexpr std::index_sequence<1UZ, std::numeric_limits<std::size_t>::max()>
      test_size{};

  "satisfy complete layer"_test = [&]<std::floating_point RealType>() {
    []<std::size_t... Size>(std::index_sequence<Size...>) {
      static_assert(
          (nou::complete_layer<nou::input_layer<RealType, Size>> && ...));
    }(test_size);
  } | test_value;

  "input size == output size"_test = [&]<std::floating_point RealType>() {
    []<std::size_t... Size>(std::index_sequence<Size...>) {
      static_assert(((nou::input_layer<RealType, Size>::input_size ==
                      nou::input_layer<RealType, Size>::output_size) &&
                     ...));
    }(test_size);
  } | test_value;
}
