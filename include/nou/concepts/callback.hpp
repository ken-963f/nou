#pragma once

#include <concepts>

#include "nou/core/error.hpp"

namespace nou {

template <class T, class U>
concept callback = std::floating_point<U> && requires(T t, U u) {
  { t.on_batch_end(u) } -> std::same_as<void>;
  { t.on_epoch_end(std::size_t{}) } -> std::same_as<void>;
  { t.on_error(error{}) } -> std::same_as<void>;
};

}
