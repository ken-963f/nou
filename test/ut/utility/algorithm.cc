#include "nou/utility/algorithm.hpp"

#include <boost/ut.hpp>
#include <execution>
#include <functional>

#include "nou/concepts/execution_policy.hpp"

auto main() -> int {
  using namespace boost::ut;

  constexpr std::tuple parallels{std::execution::par, std::execution::unseq,
                                 std::execution::par_unseq};

  "for_each"_test = [&]() {
    constexpr std::array test_value{1, 2};
    constexpr int expected = 3;

    "sequenced"_test = [&]() {
      constexpr auto result = [&]() {
        int result{};
        nou::for_each(std::execution::seq, test_value,
                      [&](auto value) { result += value; });
        return result;
      }();
      static_assert(result == expected);
    };

    "parallel"_test = [&]<nou::execution_policy P>(const P& policy) {
      auto result = [&]() {
        int result{};
        nou::for_each(policy, test_value, [&](auto value) { result += value; });
        return result;
      }();
      expect(eq(result, expected));
    } | parallels;
  };

  "transform"_test = [&]() {
    constexpr std::array test_value1{0, 1};
    constexpr std::array test_value2{2, 3};
    constexpr std::array expected{2, 4};

    "sequenced"_test = [&]() {
      constexpr auto result = [&]() {
        std::array<int, 2> result{};
        nou::transform(std::execution::seq, test_value1, test_value2,
                       result.begin(), std::plus<>{});
        return result;
      }();
      static_assert(result[0] == expected[0]);
      static_assert(result[1] == expected[1]);
    };

    "parallel"_test = [&]<nou::execution_policy P>(const P& policy) {
      auto result = [&]() {
        std::array<int, 2> result{};
        nou::transform(policy, test_value1, test_value2, result.begin(),
                       std::plus<int>{});
        return result;
      }();
      expect(eq(result[0], expected[0]));
      expect(eq(result[1], expected[1]));
    } | parallels;
  };
}
