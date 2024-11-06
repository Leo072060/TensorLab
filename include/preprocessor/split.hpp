#ifndef SPLIT_HPP
#define SPLIT_HPP

#include <algorithm>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "mat/mat.hpp"

namespace TL
{
template <typename T, typename U>
std::map<std::string, Mat<T>> train_test_split(const Mat<T> &x, const Mat<U> &y, double test_size)
{
    using namespace std;

    if (x.size(Axis::row) != y.size(Axis::row))
    {
        std::cerr << "Error: The number of rows in x and y must be the same." << std::endl;
        throw std::invalid_argument("Dimension mismatch.");
    }
    if (test_size < 0.0 || test_size > 1.0)
    {
        std::cerr << "Error: test_size must be between 0.0 and 1.0." << std::endl;
        throw std::invalid_argument("Invalid test size.");
    }

    size_t total_samples = x.size(Axis::row);
    size_t test_samples  = static_cast<size_t>(total_samples * test_size);
    size_t train_samples = total_samples - test_samples;

    vector<size_t> indices(total_samples);
    iota(indices.begin(), indices.end(), 0);

    random_device rd;
    mt19937       g(rd());
    shuffle(indices.begin(), indices.end(), g);

    Mat<T> x_train(train_samples, x.size(Axis::col));
    Mat<U> y_train(train_samples, 1);
    Mat<T> x_test(test_samples, x.size(Axis::col));
    Mat<U> y_test(test_samples, 1);

    for (size_t r = 0; r < train_samples; ++r)
    {
        for (size_t c = 0; c < x.size(Axis::col); ++c)
        {
            x_train.iloc(r, c)              = x.iloc(indices[r], c);
            x_train.iloc_name(r, Axis::row) = x.iloc_name(indices[r], Axis::row);
        }
        for (size_t c = 0; c < y.size(Axis::col); ++c)
        {
            y_train.iloc(r, c)              = y.iloc(indices[r], c);
            y_train.iloc_name(r, Axis::row) = y.iloc_name(indices[r], Axis::row);
        }
    }
    for (size_t r = 0; r < test_samples; ++r)
    {
        for (size_t c = 0; c < x.size(Axis::col); ++c)
        {
            x_test.iloc(r, c)              = x.iloc(indices[r], c);
            x_test.iloc_name(r, Axis::row) = x.iloc_name(indices[r], Axis::row);
        }
        for (size_t c = 0; c < y.size(Axis::col); ++c)
        {
            y_test.iloc(r, c)              = y.iloc(indices[r], c);
            y_test.iloc_name(r, Axis::row) = y.iloc_name(indices[r], Axis::row);
        }
    }
    for (size_t c = 0; c < x.size(Axis::col); ++c)
    {
        x_train.iloc_name(c, Axis::col) = x.iloc_name(c, Axis::col);
        x_test.iloc_name(c, Axis::col)  = x.iloc_name(c, Axis::col);
    }
    for (size_t c = 0; c < y.size(Axis::col); ++c)
    {
        y_train.iloc_name(c, Axis::col) = y.iloc_name(c, Axis::col);
        y_test.iloc_name(c, Axis::col)  = y.iloc_name(c, Axis::col);
    }

    return {{"x_train", x_train}, {"y_train", y_train}, {"x_test", x_test}, {"y_test", y_test}};
}
} // namespace TL
#endif // SPLIT_HPP