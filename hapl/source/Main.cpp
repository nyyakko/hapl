#include <Eigen/Eigen>
#include <fmt/format.h>

#include <random>
#include <vector>
#include <cassert>
#include <numeric>
#include <ranges>

using DataRow = std::vector<double>;
using DataSet = std::vector<DataRow>;

using namespace std::literals;

std::string data_row_as_string(DataRow const& dataRow)
{
    std::string result {};
    result += "[";
    for (auto separator = ""sv; auto const& data : dataRow)
    {
        result += fmt::format("{}{}", separator, data);
        separator = ", ";
    }
    result += "]";
    return result;
}

std::string data_as_string(DataSet const& dataSet)
{
    std::string result {};
    result += "[";
    for (auto separator = ""sv; auto const& row : dataSet)
    {
        result += fmt::format("{}{}", separator, data_row_as_string(row));
        separator = ", ";
    }
    result += "]";
    return result;
}

double predict(DataRow const& row, DataRow weights)
{
    auto const fnSum = [&weights, index = 1llu] (auto total, auto current) mutable {
        return total + (weights[index++] * current);
    };

    return std::accumulate(row.begin(), row.end(), weights.front(), fnSum) >= 0.0 ? 1.0 : 0.0;
}

DataRow train(DataSet const& trainingData, double learningRate, size_t epochs)
{
    DataRow weights(trainingData.front().size());

    for (auto epoch : std::views::iota(0llu, epochs))
    {
        double summedError {};

        for (auto const& row : trainingData)
        {
            auto prediction = predict(row, weights);
            auto error = row.back() - prediction;
            summedError += std::pow(error, 2.0);

            // weights.front() = weights.front() + learningRate * error;
            weights.front() = weights.front() + RAND_MAX/std::rand() * error;

            for (auto index : std::views::iota(0llu, row.size() - 1))
            {
                weights.at(index + 1) = weights.at(index + 1) + learningRate * error * row.at(index);
            }
        }

        fmt::println("epoch: {}, learning rate: {:3}, error: {:3}", epoch, learningRate, summedError);

        if (summedError < 0.5)
        {
            return weights;
        }
    }

    return weights;
}

void perceptron(std::string_view name, DataSet const& features)
{
    fmt::println("--------------- [{}] ---------------", name);

    fmt::println("training data: {}\n", data_as_string(features));

    constexpr auto learningRate = 0.1;
    constexpr auto epochs = 1000;
    const auto weights = train(features, learningRate, epochs);

    fmt::println("\ncalculated weights: {}", data_row_as_string(weights));
    fmt::println("\ntest: ");

    const DataSet input {{
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1},
    }};

    for (auto const& row : input)
    {
        fmt::println("{}: {}", data_row_as_string(row), predict(row, weights));
    }

    fmt::print("\n");
}

int main()
{
    std::srand(static_cast<uint>(time(nullptr)));

    perceptron("and", {{0,0,0}, {0,1,0}, {1,0,0}, {1,1,1}});
    // perceptron("or", {{0,0,0}, {0,1,1}, {1,0,1}, {1,1,1}});
    // perceptron("not", {{0,0,1}, {0,1,0}, {1,0,0}, {1,1,0}});
}
