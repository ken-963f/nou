#pragma once

#include <concepts>

namespace nou {

template <class T>
concept activation_function = std::copyable<T> && requires(T t) {
  { t.f(0.0F) } -> std::same_as<float>;
  { t.f(0.0) } -> std::same_as<double>;
  { t.f(0.0L) } -> std::same_as<long double>;
  { t.df(0.0F) } -> std::same_as<float>;
  { t.df(0.0) } -> std::same_as<double>;
  { t.df(0.0L) } -> std::same_as<long double>;
};

}  // namespace nou
