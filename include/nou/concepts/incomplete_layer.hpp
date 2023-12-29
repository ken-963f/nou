#pragma once

#include <concepts>
#include <type_traits>

#include "nou/concepts/complete_layer.hpp"

namespace nou {

template <class T>
concept incomplete_layer =
    // Must not satisfy complete_layer
    !complete_layer<T> &&
    // Must not throw exception at default constructor/destructor/move
    std::is_nothrow_default_constructible_v<T> &&
    std::is_nothrow_destructible_v<T> &&
    std::is_nothrow_move_constructible_v<T> &&
    std::is_nothrow_move_assignable_v<T> &&
    // Must define size_type  as member type and
    // output_size as static member variable
    std::same_as<typename T::size_type,
                 std::remove_cvref_t<decltype(T::output_size)>>;

}  // namespace nou
