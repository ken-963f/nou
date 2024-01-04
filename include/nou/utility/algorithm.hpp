#pragma once

#include <algorithm>
#include <concepts>
#include <execution>
#include <iterator>
#include <ranges>
#include <type_traits>

#include "nou/concepts/execution_policy.hpp"

namespace nou {

template <std::ranges::input_range R,
          std::invocable<std::ranges::range_value_t<R>> F>
constexpr auto for_each(const std::execution::sequenced_policy& /**/, R&& r,
                        F&& func) {
  return std::ranges::for_each(std::forward<R>(r), std::forward<F>(func));
}

template <execution_policy P, std::ranges::random_access_range R,
          std::invocable<std::ranges::range_value_t<R>> F>
auto for_each(const P& policy, R&& r, F&& func) {
  return std::for_each(policy, std::ranges::begin(r), std::ranges::end(r),
                       std::forward<F>(func));
}

template <std::ranges::input_range R1, std::ranges::input_range R2,
          std::weakly_incrementable O,
          std::invocable<std::ranges::range_value_t<R1>,
                         std::ranges::range_value_t<R2>>
              F>
  requires std::indirectly_writable<
      O, std::invoke_result_t<F, std::ranges::range_value_t<R1>,
                              std::ranges::range_value_t<R2>>>
constexpr auto transform(const std::execution::sequenced_policy& /**/, R1&& r1,
                         R2&& r2, O result, F&& func) {
  return std::ranges::transform(std::forward<R1>(r1), std::forward<R2>(r2),
                                result, std::forward<F>(func));
}

template <execution_policy P, std::ranges::random_access_range R1,
          std::ranges::random_access_range R2, std::weakly_incrementable O,
          std::invocable<std::ranges::range_value_t<R1>,
                         std::ranges::range_value_t<R2>>
              F>
  requires std::indirectly_writable<
      O, std::invoke_result_t<F, std::ranges::range_value_t<R1>,
                              std::ranges::range_value_t<R2>>>
auto transform(const P& policy, R1&& r1, R2&& r2, O result, F&& func) {
  return std::transform(policy, std::ranges::begin(r1), std::ranges::end(r1),
                        std::ranges::begin(r2), result, std::forward<F>(func));
}

}  // namespace nou
