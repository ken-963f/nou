#pragma once

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <ranges>
#include <type_traits>
#include <utility>

namespace nou {

template <std::ranges::viewable_range Input,
          std::ranges::viewable_range Teacher,
          std::invocable<std::size_t, std::size_t> F,
          std::size_t Size = std::ranges::size(Input{})>
  requires std::same_as<std::invoke_result_t<F, std::size_t, std::size_t>,
                        std::size_t> &&
           std::ranges::random_access_range<
               std::ranges::range_value_t<Input>> &&
           std::ranges::random_access_range<std::ranges::range_value_t<Teacher>>
class training_data_set final {
 public:
  [[nodiscard]] explicit constexpr training_data_set(
      const Input& input, const Teacher& teacher, F func,
      std::size_t batch_size) noexcept
      : input_{input},
        teacher_{teacher},
        function_{std::move(func)},
        batch_size_{batch_size} {}

  [[nodiscard]] constexpr auto training_data() const {
    auto data = [this]<std::size_t... I>(std::index_sequence<I...>) {
      return std::array{std::pair{std::views::all(input_[I]),
                                  std::views::all(teacher_[I])}...};
    }(std::make_index_sequence<Size>{});

    shuffule(data);

    return std::move(data) | std::views::chunk(batch_size_);
  }

 private:
  // Member variables
  std::ranges::ref_view<const Input> input_;
  std::ranges::ref_view<const Teacher> teacher_;
  F function_;
  std::size_t batch_size_{};

  // Member functions
  template <std::ranges::viewable_range R>
  constexpr auto shuffule(R& range) const {
    auto first = std::ranges::begin(range);
    auto last = std::ranges::end(range);

    for (auto it = first + 1; it != last; ++it) {
      std::iter_swap(it, first + function_(0, it - first));
    }
  }
};

}  // namespace nou
