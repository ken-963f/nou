#pragma once

#include <concepts>
#include <type_traits>

namespace nou {

template <class T>
concept complete_layer =
    // Must not throw exception at default constructor/destructor/move
    std::is_nothrow_default_constructible_v<T> &&
    std::is_nothrow_destructible_v<T> &&
    std::is_nothrow_move_constructible_v<T> &&
    std::is_nothrow_move_assignable_v<T> &&
    // Must define real_type/size_type as member type and
    // input_size/output_size as static member variable
    std::floating_point<typename T::real_type> &&
    std::same_as<typename T::size_type,
                 std::remove_cvref_t<decltype(T::input_size)>> &&
    std::same_as<typename T::size_type,
                 std::remove_cvref_t<decltype(T::output_size)>>;

}  // namespace nou
