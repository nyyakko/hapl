#include <Eigen/Eigen>
#include <fmt/format.h>
#include <fplus/fplus.hpp>
#include <raylib.h>

#include <fplus/pairs.hpp>
#include <iostream>
#include <random>
#include <vector>
#include <cassert>
#include <numeric>
#include <ranges>

using namespace std::literals;

int main()
{
    Eigen::Matrix<float, 3, 4> inputs {
        { 1, 2, 3, 2.5f },
        { 2, 5, -1, 2 },
        { -1.5, 2.7f, 3.3f, -0.8f }
    };

    Eigen::Matrix<float, 1, 3> biases1 { 2.f, 3.f, 0.5f };
    Eigen::Matrix<float, 3, 4> weights1 {
        { 0.2f, 0.8f, -0.5f, 1.f }, // + bias = neuron 1
        { 0.5f, -0.91f, 0.26f, -0.5f }, // + bias = neuron 2
        { -0.26f, -0.27f, 0.17f, 0.87f } // + bias = neuron 3
    };

    auto const layerOutputs1 = (inputs * weights1.transpose()).rowwise() + biases1;
    std::cout << layerOutputs1 << '\n';

    Eigen::Matrix<float, 1, 3> biases2 { -1.f, 2.f, -0.5f };
    Eigen::Matrix<float, 3, 3> weights2 {
        { 0.1f, -0.14f, 0.5f }, // + bias = neuron 1
        { -0.5f, 0.12f, -0.33f }, // + bias = neuron 2
        { -0.44f, 0.73f, -0.13f } // + bias = neuron 3
    };

    auto const layerOutputs2 = (layerOutputs1 * weights2.transpose()).rowwise() + biases2;
    std::cout << layerOutputs2 << '\n';
}
