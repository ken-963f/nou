#pragma once

#include <concepts>
#include <ranges>

namespace nou {

template <class T, class U>
concept metric =
    std::ranges::random_access_range<U> &&
    std::floating_point<std::ranges::range_value_t<U>> &&
    std::invocable<T, U, U> &&
    std::same_as<std::ranges::range_value_t<U>, std::invoke_result_t<T, U, U>>;

}
