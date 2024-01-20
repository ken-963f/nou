#include "nou/network.hpp"

#include <concepts>
#include <execution>
#include <expected>
#include <optional>
#include <ranges>
#include <string_view>
#include <tuple>
#include <type_traits>

#include "boost/ut.hpp"
#include "nou/concepts/execution_policy.hpp"
#include "nou/core/error.hpp"
#include "nou/core/training_data_set.hpp"
#include "nou/layer/input_layer.hpp"

namespace mock {

template <std::floating_point RealType, std::size_t InputSize,
          std::size_t OutputSize>
struct complete_layer final {
  // Member Type
  using real_type = RealType;
  using size_type = std::size_t;
  using value_type = real_type;
  using input_type = std::array<real_type, 1>;
  using output_type = std::array<real_type, 1>;
  using backward_type = output_type;
  using loss_type = output_type;

  // Static Member
  static constexpr size_type input_size = InputSize;
  static constexpr size_type output_size = OutputSize;

  // Constructor
  complete_layer() = default;
  [[nodiscard]] explicit constexpr complete_layer(
      real_type value,
      std::optional<std::unexpected<nou::error>> forward_propagation_error =
          std::nullopt,
      std::optional<std::unexpected<nou::error>> backward_propagation_error =
          std::nullopt)
      : value_{value},
        forward_propagation_error_{std::move(forward_propagation_error)},
        backward_propagation_error_{std::move(backward_propagation_error)} {}

  // Public Function
  [[nodiscard]] constexpr auto forward_propagate(
      const nou::execution_policy auto& /**/,
      std::span<const real_type, 1> value) const noexcept
      -> std::expected<output_type, nou::error> {
    if (forward_propagation_error_.has_value()) {
      return forward_propagation_error_.value();
    } else {
      return std::array{real_type{value[0] + value_}};
    }
  }

  [[nodiscard]] constexpr auto backward_propagate(
      const nou::execution_policy auto& /**/,
      std::span<const real_type, 1> output,
      std::span<const real_type, 1> loss) const noexcept
      -> std::expected<backward_type, nou::error> {
    if (backward_propagation_error_.has_value()) {
      return backward_propagation_error_.value();
    } else {
      return std::array{real_type{output[0] + loss[0]}};
    }
  }

  constexpr void add_gradient(const nou::execution_policy auto& /**/,
                              std::span<const real_type, 1> output,
                              std::span<const real_type, 1> loss) noexcept {
    update_ += output[0] + loss[0];
  }

  constexpr void apply_gradient(const nou::execution_policy auto& /**/) {
    value_ += update_;
  }

  // Getter/Setter
  [[nodiscard]] constexpr auto value() const noexcept -> value_type {
    return value_;
  }

 private:
  value_type value_{};
  value_type update_{};
  std::optional<std::unexpected<nou::error>> forward_propagation_error_{};
  std::optional<std::unexpected<nou::error>> backward_propagation_error_{};
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

struct loss_function final {
  template <std::floating_point RealType>
  constexpr auto f(RealType output, RealType teacher) const noexcept
      -> RealType {
    return output + teacher;
  }

  template <std::floating_point RealType>
  constexpr auto df(RealType output, RealType teacher) const noexcept
      -> RealType {
    return output + teacher;
  }
};

template <std::floating_point RealType>
struct callback final {
  RealType metric{};
  std::size_t epoch_sum{};
  nou::error error{};

  constexpr void on_batch_end(RealType x) noexcept { metric += x; }
  constexpr void on_epoch_end(std::size_t x) noexcept { epoch_sum += x; }
  constexpr void on_error(const nou::error& x) noexcept { error = x; }
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
    static_assert(std::get<0>(value1) == RealType{});
    static_assert(std::get<1>(value1) == RealType{});

    constexpr auto value2 = nou::network{
        nou::input_layer<RealType, 1UZ>{},
        mock::complete_layer<RealType, 1UZ, 2UZ>{1.0},
        mock::complete_layer<RealType, 2UZ, 3UZ>{
            2.0}}.value();
    static_assert(std::get<0>(value2) == RealType{1.0});
    static_assert(std::get<1>(value2) == RealType{2.0});
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

  constexpr std::unexpected error{nou::error{.what = error_message}};

  "predict_error"_test = [&]<std::floating_point RealType>() {
    constexpr auto input = std::array{RealType{1.0}};
    {
      constexpr auto network =
          nou::network{nou::input_layer<RealType, 1UZ>{},
                       mock::complete_layer<RealType, 1UZ, 2UZ>{1.0, error},
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
                       mock::complete_layer<RealType, 2UZ, 3UZ>{1.0, error}};
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
                       mock::complete_layer<RealType, 1UZ, 2UZ>{1.0, error},
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
                       mock::complete_layer<RealType, 2UZ, 3UZ>{1.0, error}};

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

  "fit"_test = [&]<std::floating_point RealType>() {
    static constexpr std::array<std::array<RealType, 1>, 1> batch_input{
        std::array{RealType{1.0}}};
    static constexpr std::array<std::array<RealType, 3>, 1> batch_teacher{
        std::array{RealType{5.0}, RealType{}, RealType{}}};

    constexpr auto shuffule_func = [](auto /**/, auto /**/) { return 0UZ; };
    constexpr nou::training_data_set data_set{batch_input, batch_teacher,
                                              shuffule_func, 1UZ};
    constexpr mock::metric metric{};
    constexpr mock::loss_function loss_function{};
    constexpr auto network =
        nou::network{nou::input_layer<RealType, 1UZ>{},
                     mock::complete_layer<RealType, 1UZ, 2UZ>{1.0},
                     mock::complete_layer<RealType, 2UZ, 3UZ>{2.0}};

    "sequenced execution"_test = [&]() {
      constexpr auto tuple = [&]() {
        auto network1 = nou::network(network);
        mock::callback<RealType> callback{};
        network1.fit(data_set, loss_function, metric, callback, 1UZ);
        auto value1 = callback.metric;
        auto value2 = network1.value();
        return std::tuple{value1, value2};
      }();
      constexpr auto value1 = std::get<0>(tuple);
      constexpr auto value2 = std::get<1>(tuple);

      static_assert(value1 == RealType{4.0 - 5.0});

      static_assert(std::get<0>(value2) ==
                    RealType{1.0 + (4.0 + 5.0) + 4.0 + 2.0});
      static_assert(std::get<1>(value2) == RealType{2.0 + (4.0 + 5.0) + 4.0});
    };

    "parallel execution"_test = [&, network =
                                        nou::network(network)](auto&& policy) {
      auto [value1, value2] = [&]() {
        auto network1 = nou::network(network);
        mock::callback<RealType> callback{};
        network1.fit(policy, data_set, loss_function, metric, callback, 1UZ);
        auto value1 = callback.metric;
        auto value2 = network1.value();
        return std::tuple{value1, value2};
      }();

      expect(eq(value1, RealType{4.0 - 5.0}));

      expect(eq(std::get<0>(value2), RealType{1.0 + (4.0 + 5.0) + 4.0 + 2.0}));
      expect(eq(std::get<1>(value2), RealType{2.0 + (4.0 + 5.0) + 4.0}));
    } | policies;
  } | test_value;

  "fit error"_test = [&]<std::floating_point RealType>() {
    static constexpr std::array<std::array<RealType, 1>, 1> batch_input{
        std::array{RealType{1.0}}};
    static constexpr std::array<std::array<RealType, 3>, 1> batch_teacher{
        std::array{RealType{5.0}, RealType{}, RealType{}}};

    constexpr auto shuffule_func = [](auto /**/, auto /**/) { return 0UZ; };
    constexpr nou::training_data_set data_set{batch_input, batch_teacher,
                                              shuffule_func, 1UZ};
    constexpr mock::metric metric{};
    constexpr mock::loss_function loss_function{};

    {
      constexpr auto network =
          nou::network{nou::input_layer<RealType, 1UZ>{},
                       mock::complete_layer<RealType, 1UZ, 2UZ>{1.0, error},
                       mock::complete_layer<RealType, 2UZ, 3UZ>{2.0}};

      "sequenced execution"_test = [&]() {
        constexpr auto tuple = [&]() {
          auto network1 = nou::network(network);
          mock::callback<RealType> callback{};
          network1.fit(data_set, loss_function, metric, callback, 1UZ);
          auto value1 = callback.error;
          auto value2 = network1.value();
          return std::tuple{value1, value2};
        }();
        constexpr auto value1 = std::get<0>(tuple);
        constexpr auto value2 = std::get<1>(tuple);

        static_assert(value1.what == error_message);
        static_assert(value1.layer_index == 0);

        static_assert(std::get<0>(value2) == RealType{1.0});
        static_assert(std::get<1>(value2) == RealType{2.0});
      };

      "parallel execution"_test = [&, network = nou::network(network)](
                                      auto&& policy) {
        auto [value1, value2] = [&]() {
          auto network1 = nou::network(network);
          mock::callback<RealType> callback{};
          network1.fit(policy, data_set, loss_function, metric, callback, 1UZ);
          auto value1 = callback.error;
          auto value2 = network1.value();
          return std::tuple{value1, value2};
        }();

        expect(eq(value1.what, error_message));
        expect(eq(value1.layer_index, 0));

        expect(eq(std::get<0>(value2), RealType{1.0}));
        expect(eq(std::get<1>(value2), RealType{2.0}));
      } | policies;
    }
    {
      constexpr auto network =
          nou::network{nou::input_layer<RealType, 1UZ>{},
                       mock::complete_layer<RealType, 1UZ, 2UZ>{1.0},
                       mock::complete_layer<RealType, 2UZ, 3UZ>{2.0, error}};

      "sequenced execution"_test = [&]() {
        constexpr auto tuple = [&]() {
          auto network1 = nou::network(network);
          mock::callback<RealType> callback{};
          network1.fit(data_set, loss_function, metric, callback, 1UZ);
          auto value1 = callback.error;
          auto value2 = network1.value();
          return std::tuple{value1, value2};
        }();
        constexpr auto value1 = std::get<0>(tuple);
        constexpr auto value2 = std::get<1>(tuple);

        static_assert(value1.what == error_message);
        static_assert(value1.layer_index == 1);

        static_assert(std::get<0>(value2) == RealType{1.0});
        static_assert(std::get<1>(value2) == RealType{2.0});
      };

      "parallel execution"_test = [&, network = nou::network(network)](
                                      auto&& policy) {
        auto [value1, value2] = [&]() {
          auto network1 = nou::network(network);
          mock::callback<RealType> callback{};
          network1.fit(policy, data_set, loss_function, metric, callback, 1UZ);
          auto value1 = callback.error;
          auto value2 = network1.value();
          return std::tuple{value1, value2};
        }();

        expect(eq(value1.what, error_message));
        expect(eq(value1.layer_index, 1));

        expect(eq(std::get<0>(value2), RealType{1.0}));
        expect(eq(std::get<1>(value2), RealType{2.0}));
      } | policies;
    }
  } | test_value;
}
