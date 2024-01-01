#include "nou/network.hpp"

#include <concepts>
#include <execution>
#include <expected>
#include <ranges>
#include <string_view>
#include <tuple>
#include <type_traits>

#include "boost/ut.hpp"
#include "nou/concepts/execution_policy.hpp"
#include "nou/core/error.hpp"
#include "nou/layer/input_layer.hpp"

namespace mock {

template <std::floating_point RealType, std::size_t InputSize,
          std::size_t OutputSize>
struct complete_layer final {
  // Member Type
  using real_type = RealType;
  using size_type = std::size_t;
  using value_type = std::expected<real_type, nou::error>;
  using output_type = std::expected<std::array<real_type, 1>, nou::error>;

  // Static Member
  static constexpr size_type input_size = InputSize;
  static constexpr size_type output_size = OutputSize;

  // Constructor
  complete_layer() = default;
  [[nodiscard]] explicit constexpr complete_layer(real_type value)
      : value_{value} {}
  [[nodiscard]] explicit constexpr complete_layer(nou::error error)
      : value_{std::unexpected{error}} {}

  // Public Function
  [[nodiscard]] constexpr auto forward_propagate(
      const nou::execution_policy auto& policy,
      std::span<const real_type, 1> value) const noexcept -> output_type {
    return value_.transform([value = std::move(value)](auto value_) {
      return std::array{real_type{value[0] + value_}};
    });
  }

  // Getter/Setter
  [[nodiscard]] constexpr auto value() const noexcept -> value_type {
    return value_;
  }

 private:
  value_type value_{};
};

template <std::size_t OutputSize>
struct incomplete_layer final {
  // Member Type
  using size_type = std::size_t;

  // Static Member
  static constexpr size_type output_size = OutputSize;

  // Public Functions
  template <nou::complete_layer PrevLayer>
  constexpr auto make_complete_layer() noexcept {
    return complete_layer<typename PrevLayer::real_type, PrevLayer::output_size,
                          OutputSize>{};
  }
};

struct metric final {
  template <std::ranges::random_access_range Output,
            std::ranges::random_access_range Teacher>
  constexpr auto operator()(Output&& output, Teacher&& teacher) const {
    return output[0] - teacher[0];
  }
};

}  // namespace mock

auto main() -> int {
  using namespace boost::ut;

  constexpr std::tuple<float, double, long double> test_value = {};
  constexpr auto policies =
      std::tuple{std::execution::seq, std::execution::par,
                 std::execution::unseq, std::execution::par_unseq};
  constexpr std::string_view error_message = "test";

  "default constructor"_test = []<std::floating_point RealType>() {
    static_assert(
        std::is_nothrow_default_constructible_v<nou::network<
            nou::input_layer<RealType, 1UZ>, mock::incomplete_layer<2UZ>>>);
    static_assert(
        std::is_nothrow_default_constructible_v<nou::network<
            nou::input_layer<RealType, 1UZ>, mock::incomplete_layer<2UZ>,
            mock::incomplete_layer<3UZ>>>);
  } | test_value;

  "constructor"_test = []<std::floating_point RealType>() {
    constexpr auto value1 = nou::network{nou::input_layer<RealType, 1UZ>{},
                                         mock::incomplete_layer<2UZ>{},
                                         mock::incomplete_layer<3UZ>{}}
                                .value();
    static_assert(std::get<0>(value1).has_value());
    static_assert(std::get<0>(value1).value() == RealType{});
    static_assert(std::get<1>(value1).has_value());
    static_assert(std::get<1>(value1).value() == RealType{});

    constexpr auto value2 = nou::network{
        nou::input_layer<RealType, 1UZ>{},
        mock::complete_layer<RealType, 1UZ, 2UZ>{1.0},
        mock::complete_layer<RealType, 2UZ, 3UZ>{
            2.0}}.value();
    static_assert(std::get<0>(value2).has_value());
    static_assert(std::get<0>(value2).value() == RealType{1.0});
    static_assert(std::get<1>(value2).has_value());
    static_assert(std::get<1>(value2).value() == RealType{2.0});
  } | test_value;

  "predict"_test = [&]<std::floating_point RealType>() {
    constexpr auto input = std::array{RealType{1.0}};
    constexpr auto network =
        nou::network{nou::input_layer<RealType, 1UZ>{},
                     mock::complete_layer<RealType, 1UZ, 2UZ>{1.0},
                     mock::complete_layer<RealType, 2UZ, 3UZ>{2.0}};
    constexpr auto value = network.predict(input);
    static_assert(value.has_value());
    static_assert(value.value() == std::array{RealType{4.0}});

    std::apply(
        [&network, &input](auto&&... policies) {
          (
              [&network, &input](auto&& policy) {
                auto value = network.predict(policy, input);
                expect(value.has_value());
                expect(eq(value.value(), std::array{RealType{4.0}}));
              }(policies),
              ...);
        },
        policies);
  } | test_value;

  "predict_error"_test = [&]<std::floating_point RealType>() {
    constexpr auto input = std::array{RealType{1.0}};
    {
      constexpr auto network =
          nou::network{nou::input_layer<RealType, 1UZ>{},
                       mock::complete_layer<RealType, 1UZ, 2UZ>{
                           nou::error{.what = error_message}},
                       mock::complete_layer<RealType, 2UZ, 3UZ>{1.0}};
      constexpr auto value = network.predict(input);
      static_assert(!value.has_value());
      static_assert(value.error().what == error_message);
      static_assert(value.error().layer_index == 0);

      std::apply(
          [&](auto&&... policies) {
            (
                [&](auto&& policy) {
                  auto value = network.predict(policy, input);
                  expect(!value.has_value());
                  expect(eq(value.error().what, error_message));
                  expect(eq(value.error().layer_index, 0));
                }(policies),
                ...);
          },
          policies);
    }
    {
      constexpr auto network =
          nou::network{nou::input_layer<RealType, 1UZ>{},
                       mock::complete_layer<RealType, 1UZ, 2UZ>{1.0},
                       mock::complete_layer<RealType, 2UZ, 3UZ>{
                           nou::error{.what = error_message}}};
      constexpr auto value = network.predict(input);
      static_assert(!value.has_value());
      static_assert(value.error().what == error_message);
      static_assert(value.error().layer_index == 1);

      std::apply(
          [&](auto&&... policies) {
            (
                [&](auto&& policy) {
                  auto value = network.predict(policy, input);
                  expect(!value.has_value());
                  expect(eq(value.error().what, error_message));
                  expect(eq(value.error().layer_index, 1));
                }(policies),
                ...);
          },
          policies);
    }
  } | test_value;

  "evaluate"_test = [&]<std::floating_point RealType>() {
    constexpr auto input = std::array{RealType{1.0}};
    constexpr auto teacher = std::array{RealType{5.0}, RealType{}, RealType{}};
    constexpr mock::metric metric{};
    constexpr auto network =
        nou::network{nou::input_layer<RealType, 1UZ>{},
                     mock::complete_layer<RealType, 1UZ, 2UZ>{1.0},
                     mock::complete_layer<RealType, 2UZ, 3UZ>{2.0}};

    "sequenced execution"_test = [&]() {
      constexpr auto value = network.evaluate(input, teacher, metric);
      static_assert(value.has_value());
      static_assert(value.value() == RealType{4.0 - 5.0});
    };

    "parallel execution"_test = [&](auto&& policy) {
      auto value = network.evaluate(policy, input, teacher, metric);
      expect(value.has_value());
      expect(eq(value.value(), RealType{4.0 - 5.0}));
    } | policies;
  } | test_value;

  "evaluate error"_test = [&]<std::floating_point RealType>() {
    constexpr auto input = std::array{RealType{1.0}};
    constexpr auto teacher = std::array{RealType{5.0}, RealType{}, RealType{}};
    constexpr mock::metric metric{};
    {
      constexpr auto network =
          nou::network{nou::input_layer<RealType, 1UZ>{},
                       mock::complete_layer<RealType, 1UZ, 2UZ>{
                           nou::error{.what = error_message}},
                       mock::complete_layer<RealType, 2UZ, 3UZ>{2.0}};

      "sequenced execution"_test = [&]() {
        constexpr auto value = network.evaluate(input, teacher, metric);
        static_assert(!value.has_value());
        static_assert(value.error().what == error_message);
        static_assert(value.error().layer_index == 0);
      };

      "parallel execution"_test = [&](auto&& policy) {
        auto value = network.evaluate(policy, input, teacher, metric);
        expect(!value.has_value());
        expect(eq(value.error().what, error_message));
        expect(eq(value.error().layer_index, 0));
      } | policies;
    }
    {
      constexpr auto network =
          nou::network{nou::input_layer<RealType, 1UZ>{},
                       mock::complete_layer<RealType, 1UZ, 2UZ>{1.0},
                       mock::complete_layer<RealType, 2UZ, 3UZ>{
                           nou::error{.what = error_message}}};

      "sequenced execution"_test = [&]() {
        constexpr auto value = network.evaluate(input, teacher, metric);
        static_assert(!value.has_value());
        static_assert(value.error().what == error_message);
        static_assert(value.error().layer_index == 1);
      };

      "parallel execution"_test = [&](auto&& policy) {
        auto value = network.evaluate(policy, input, teacher, metric);
        expect(!value.has_value());
        expect(eq(value.error().what, error_message));
        expect(eq(value.error().layer_index, 1));
      } | policies;
    }
  } | test_value;
}
