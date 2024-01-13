#include "nou/core/training_data_set.hpp"

#include <boost/ut.hpp>
#include <concepts>

auto main() -> int {
  using namespace boost::ut;

  "training_data"_test = []<std::floating_point RealType> {
    constexpr auto training_data1 = []() {
      static constexpr std::array<std::array<RealType, 1>, 2> batch_input{
          std::array{RealType{1.0}}, std::array{RealType{2.0}}};
      static constexpr std::array batch_teacher{std::array{RealType{-1.0}},
                                                std::array{RealType{-2.0}}};
      constexpr auto func = [](auto, auto) { return 0UZ; };
      nou::training_data_set dataset{batch_input, batch_teacher, func, 1UZ};
      return dataset.training_data();
    }();

    static_assert(std::ranges::size(training_data1) == 2UZ);
    static_assert(std::get<0>(training_data1[0][0])[0] == RealType{2.0});
    static_assert(std::get<1>(training_data1[0][0])[0] == RealType{-2.0});
    static_assert(std::get<0>(training_data1[1][0])[0] == RealType{1.0});
    static_assert(std::get<1>(training_data1[1][0])[0] == RealType{-1.0});

    constexpr auto training_data2 = []() {
      static constexpr std::array<std::array<RealType, 1>, 2> batch_input{
          std::array{RealType{1.0}}, std::array{RealType{2.0}}};
      static constexpr std::array batch_teacher{std::array{RealType{-1.0}},
                                                std::array{RealType{-2.0}}};
      constexpr auto func = [](auto, auto) { return 0UZ; };

      constexpr auto view = batch_input | std::views::chunk(2);
      static_assert(std::ranges::size(view) == 1UZ);

      nou::training_data_set dataset{batch_input, batch_teacher, func, 2UZ};
      return dataset.training_data();
    }();

    static_assert(std::ranges::size(training_data2) == 1UZ);
    static_assert(std::get<0>(training_data2[0][0])[0] == RealType{2.0});
    static_assert(std::get<1>(training_data2[0][0])[0] == RealType{-2.0});
    static_assert(std::get<0>(training_data2[0][1])[0] == RealType{1.0});
    static_assert(std::get<1>(training_data2[0][1])[0] == RealType{-1.0});
  } | std::tuple<float, double, long double>{};
}
