#pragma once

#include <concepts>
#include <type_traits>

namespace nou {

template <class T, class U>
concept loss_function =
    std::floating_point<U> && requires(std::remove_cvref_t<T> t, U u) {
      { t.f(u, u) } -> std::same_as<U>;
      { t.df(u, u) } -> std::same_as<U>;
    };

}
