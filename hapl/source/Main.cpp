#include "Data.hpp"
#include "Math.hpp"

#include <Eigen/Eigen>
#include <fmt/format.h>
#include <fplus/fplus.hpp>
#include <raylib.h>

auto layer_dense_forward(Eigen::MatrixXd const& weights, Eigen::Matrix<double, 1, Eigen::Dynamic> const& biases, Eigen::MatrixXd const& inputs)
{
    auto const neurons = (inputs * weights).rowwise() + biases;
    return std::make_tuple(weights.eval(), biases.eval(), neurons.eval());
}

auto layer_dense_forward(Eigen::MatrixXd const& inputs, long inputsCount, long neuronsCount)
{
    auto const weights = 0.01 * Eigen::MatrixXd::Random(inputsCount, neuronsCount);
    auto const biases  = Eigen::VectorXd::Zero(neuronsCount).transpose();
    auto const neurons = (inputs * weights).rowwise() + biases;
    return std::make_tuple(weights.eval(), biases.eval(), neurons.eval());
}

auto layer_dense_backward(Eigen::MatrixXd const& weights, Eigen::MatrixXd const& dvalues, Eigen::MatrixXd const& inputs)
{
    auto const dweights = inputs.transpose() * dvalues;
    auto const dbiases  = dvalues.colwise().sum();
    auto const dinputs  = dvalues * weights.transpose();
    return std::make_tuple(dweights.eval(), dbiases.eval(), dinputs.eval());
}

auto activation_relu_forward(Eigen::MatrixXd const& outputs)
{
    return outputs.unaryExpr([] (auto x) { return std::max(0.0, x); }).eval();
}

auto activation_relu_backward(Eigen::MatrixXd const& inputs, Eigen::MatrixXd const& dinputs)
{
    Eigen::MatrixXd result = dinputs;
    for (auto i : fplus::numbers(0l, inputs.rows()))
    {
        for (auto w : fplus::numbers(0l, inputs.cols()))
            result.row(i).col(w).x() = inputs.row(i).col(w).value() <= 0 ? 0.0 : result.row(i).col(w).x();
    }
    return result;
}

auto activation_softmax_forward(Eigen::MatrixXd const& outputs)
{
    auto result = outputs;

    for (auto rowIndex : fplus::numbers(0l, outputs.rows()))
    {
         result.row(rowIndex) = result.row(rowIndex).unaryExpr([max = result.row(rowIndex).maxCoeff()] (auto x) { return std::pow(std::numbers::e, x - max); }).eval();
         result.row(rowIndex) = result.row(rowIndex).unaryExpr([sum = result.row(rowIndex).sum()] (auto x) { return x / sum; }).eval();
    }

    return result.eval();
}

auto activation_softmax_backward(Eigen::MatrixXd const& dvalues, Eigen::MatrixXd const& output)
{
    Eigen::MatrixXd dinputs = Eigen::MatrixXd::Zero(dvalues.rows(), dvalues.cols());

    for (auto i : fplus::numbers(0l, dvalues.rows()))
    {
        auto const lhs = diagflat(output.row(i).transpose());
        auto const rhs = output.row(i).transpose() * output.row(i);
        Eigen::MatrixXd jacobianMatrix = lhs - rhs;
        for (auto j : fplus::numbers(0l, jacobianMatrix.rows()))
            jacobianMatrix.row(j) = jacobianMatrix.row(j).cwiseProduct(dvalues.row(i));
        dinputs.row(i) = jacobianMatrix.rowwise().sum().transpose();
    }

    return dinputs.eval();
}

auto categorical_cross_entropy_forward(Eigen::MatrixXd const& predicted, Eigen::MatrixXd const& truth)
{
    Eigen::MatrixXd confidences {};

    if (truth.rows() == 1)
    {
        confidences = Eigen::MatrixXd::Zero(1, predicted.rows());
        for (auto const& [targetClass, distribution] :
                fplus::zip(truth.subVector<Eigen::Horizontal>(0).cast<long>(), fplus::numbers(0l, predicted.rows())))
        {
            confidences.row(0).col(distribution) = predicted.row(distribution).col(targetClass);
        }
    }
    else if (truth.rows() == 2)
    {
        confidences = (predicted * truth).rowwise().sum();
    }

    return confidences.unaryExpr([] (auto x) { return -std::log(x); }).eval();
}

auto categorical_cross_entropy_backward(Eigen::MatrixXd const& predicted, Eigen::MatrixXd truth)
{
    if (truth.rows() == 1) truth = eye(predicted.cols(), truth);
    Eigen::MatrixXd dinputs = ((-1 * truth).array() / predicted.array());
    return (dinputs / predicted.rows()).eval();
}

auto accuracy(Eigen::MatrixXd const& outputs, Eigen::MatrixXd const& targets)
{
    auto const predictions = argmax(outputs, Axis::COL);
    return predictions.cwiseEqual(targets).cast<double>().mean();
}

auto stochastic_gradient_descent(Eigen::MatrixXd weights, Eigen::MatrixXd const& dweights, Eigen::MatrixXd biases, Eigen::MatrixXd const& dbiases, double learningRate)
{
    weights += -learningRate * dweights;
    biases  += -learningRate * dbiases;
    return std::make_tuple(weights.eval(), biases.eval());
}

int main()
{
    // INPUT LAYER
    Eigen::MatrixXd X { data_g };
    Eigen::MatrixXd y { targets_g };

    // HIDDEN LAYER 1
    auto [dense1Weights, dense1Biases, dense1] = layer_dense_forward(X, 2, 64);
    auto activation1 = activation_relu_forward(dense1);

    // HIDDEN LAYER 2
    auto [dense2Weights, dense2Biases, dense2] = layer_dense_forward(activation1, 64, 3);
    auto activation2 = activation_softmax_forward(dense2);

    for (auto epoch : fplus::numbers(0l, 10001l))
    {
        if (epoch != 0)
        {
            auto const [updatedWeights1, updatedBiases1, updatedLayer1] = layer_dense_forward(dense1Weights, dense1Biases, X);
            dense1Weights = updatedWeights1;
            dense1Biases = updatedBiases1;
            dense1 = updatedLayer1;
            activation1 = activation_relu_forward(updatedLayer1);

            auto const [updatedWeights2, updatedBiases2, updatedLayer2] = layer_dense_forward(dense2Weights, dense2Biases, activation1);
            dense2Weights = updatedWeights2;
            dense2Biases = updatedBiases2;
            dense2 = updatedLayer2;
            activation2 = activation_softmax_forward(updatedLayer2);
        }

        auto const costActivation = categorical_cross_entropy_forward(activation2, y);

        if (epoch % 1000 == 0)
        {
            fmt::println("epoch: {}", epoch);
            fmt::println("cost: {}", costActivation.mean());
            fmt::println("acc: {}", accuracy(activation2, y));
            fmt::print("\n");
        }

        // BACKWARD PASS
        auto const costActivationBackward = categorical_cross_entropy_backward(activation2, y);
        auto const softmaxActivationBackward = activation_softmax_backward(costActivationBackward, activation2);
        auto const [dense2dWeights, dense2dBiases, dense2dInputs] = layer_dense_backward(dense2Weights, softmaxActivationBackward, activation1);
        auto const reluActivationBackward = activation_relu_backward(dense1, dense2dInputs);
        auto const [dense1dWeights, dense1dBiases, dense1dInputs] = layer_dense_backward(dense1Weights, reluActivationBackward, X);

        auto [updated1Weights, updated1Biases] = stochastic_gradient_descent(dense1Weights, dense1dWeights, dense1Biases, dense1dBiases, 1.0);
        dense1Weights = updated1Weights;
        dense1Biases = updated1Biases;

        auto [updated2Weights, updated2Biases] = stochastic_gradient_descent(dense2Weights, dense2dWeights, dense2Biases, dense2dBiases, 1.0);
        dense2Weights = updated2Weights;
        dense2Biases = updated2Biases;
    }
}
