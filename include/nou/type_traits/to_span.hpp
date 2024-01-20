#pragma once

#include <ranges>
#include <span>

namespace nou {

namespace detail {

template <class T>
struct to_span;

template <std::ranges::range T>
  requires(std::ranges::size(T{}) > 0)
struct to_span<T> final {
  using type = std::span<std::ranges::range_value_t<T>, std::ranges::size(T{})>;
};

template <std::ranges::range T>
struct to_span<T> final {
  using type = std::span<std::ranges::range_value_t<T>, std::dynamic_extent>;
};

template <std::ranges::range T>
  requires requires { T::extent; }
struct to_span<T> final {
  using type = std::span<std::ranges::range_value_t<T>, T::extent>;
};

template <class T>
struct to_const_span;

template <std::ranges::range T>
  requires(std::ranges::size(T{}) > 0)
struct to_const_span<T> final {
  using type =
      std::span<const std::ranges::range_value_t<T>, std::ranges::size(T{})>;
};

template <std::ranges::range T>
struct to_const_span<T> final {
  using type =
      std::span<const std::ranges::range_value_t<T>, std::dynamic_extent>;
};

template <std::ranges::range T>
  requires requires { T::extent; }
struct to_const_span<T> final {
  using type = std::span<const std::ranges::range_value_t<T>, T::extent>;
};

}  // namespace detail

template <std::ranges::range T>
using to_span_t = typename detail::to_span<T>::type;

template <std::ranges::range T>
using to_const_span_t = typename detail::to_const_span<T>::type;

};  // namespace nou
