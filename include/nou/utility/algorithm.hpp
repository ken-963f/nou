#pragma once

#include <algorithm>
#include <concepts>
#include <execution>
#include <iterator>
#include <numeric>
#include <ranges>
#include <type_traits>

#include "nou/concepts/execution_policy.hpp"

namespace nou {

template <std::ranges::input_range R,
          std::invocable<std::ranges::range_value_t<R>> F>
constexpr auto for_each(const std::execution::sequenced_policy& /**/, R&& r,
                        F func) {
  return std::ranges::for_each(std::forward<R>(r), func);
}

template <execution_policy P, std::ranges::random_access_range R,
          std::invocable<std::ranges::range_value_t<R>> F>
auto for_each(const P& policy, R&& r, F func) {
  return std::for_each(policy, std::ranges::begin(r), std::ranges::end(r),
                       func);
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
                         R2&& r2, O result, F func) {
  return std::ranges::transform(std::forward<R1>(r1), std::forward<R2>(r2),
                                result, func);
}

template <execution_policy P, std::ranges::random_access_range R1,
          std::ranges::random_access_range R2, std::weakly_incrementable O,
          std::invocable<std::ranges::range_value_t<R1>,
                         std::ranges::range_value_t<R2>>
              F>
  requires std::indirectly_writable<
      O, std::invoke_result_t<F, std::ranges::range_value_t<R1>,
                              std::ranges::range_value_t<R2>>>
auto transform(const P& policy, R1&& r1, R2&& r2, O result, F func) {
  return std::transform(policy, std::ranges::begin(r1), std::ranges::end(r1),
                        std::ranges::begin(r2), result, func);
}

template <execution_policy P, std::ranges::random_access_range R,
          std::copy_constructible T, std::invocable<T, T> BinaryOperation,
          std::invocable<std::ranges::range_value_t<R>> UnaryOperation>
  requires std::convertible_to<std::invoke_result_t<BinaryOperation, T, T>,
                               T> &&
           std::convertible_to<
               std::invoke_result_t<UnaryOperation,
                                    std::ranges::range_value_t<R>>,
               T>
constexpr auto transform_reduce(const P& policy, R&& r, T init,
                                BinaryOperation binary_op,
                                UnaryOperation unary_op) -> T {
  if constexpr (std::same_as<P, std::execution::sequenced_policy>) {
    return std::transform_reduce(std::ranges::begin(r), std::ranges::end(r),
                                 init, binary_op, unary_op);
  } else {
    return std::transform_reduce(policy, std::ranges::begin(r),
                                 std::ranges::end(r), init, binary_op,
                                 unary_op);
  }
}

}  // namespace nou
