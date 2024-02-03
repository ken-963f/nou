#pragma once

#include <concepts>
#include <type_traits>

namespace nou {

template <class T>
concept initializer =
    std::copyable<T> && std::floating_point<std::invoke_result_t<T>>;

}
