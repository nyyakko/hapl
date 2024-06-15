#pragma once

#include <Eigen/Eigen>
#include <fmt/format.h>
#include <fplus/fplus.hpp>
#include <raylib.h>

#include <vector>

enum class Axis { ROW, COL };

inline auto argmax(Eigen::MatrixXd const& matrix, Axis axis)
{
    std::vector<double> result {};

    for (auto rowIndex : fplus::numbers(0l, matrix.rows()))
    {
        long row {}; long col {};
        matrix.row(rowIndex).maxCoeff(&row, &col);
        result.push_back(static_cast<double>(axis == Axis::ROW ? row : col));
    }

    return Eigen::Map<Eigen::MatrixXd>(result.data(), 1, matrix.rows()).eval();
}

inline auto eye(long size)
{
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(size, size);
    for (auto index : fplus::numbers(0l, result.cols()))
        result.row(index).col(index).x() = 1.0;
    return result.eval();
}

inline auto eye(long size, Eigen::MatrixXd const& other)
{
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(other.cols(), size);
    for (auto index : fplus::numbers(0l, result.rows()))
        result.row(index).col(other.row(0).col(index).cast<long>().value()).x() = 1.0;
    return result.eval();
}

inline auto diagflat(Eigen::MatrixXd const& lhs)
{
    auto const rhs = eye(lhs.rows());
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(rhs.rows(), rhs.cols());
    for (auto index : fplus::numbers(0l, rhs.cols()))
        result.row(index).col(index).x() = lhs.row(index).x() * rhs.row(index).col(index).x();
    return result.eval();
}
