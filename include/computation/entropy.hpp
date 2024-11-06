#ifndef ENTROPY_HPP
#define ENTROPY_HPP

#include <map>

#include "mat/mat.hpp"

namespace TL
{
using namespace _internal;
template <typename T> double entropy(const Mat<T> &data);
template <typename T> double entropy(const Mat<T> &data, const Mat<double> &weight);

template <typename T> double entropy(const Mat<T> &data)
{
    Mat<double> weight(data.size(Axis::row), data.size(Axis::col));
    weight = 1;
    return entropy(data, weight);
}
template <typename T> double entropy(const Mat<T> &data, const Mat<double> &weight)
{
    using namespace std;

    if (data.size(Axis::row) != weight.size(Axis::row) || data.size(Axis::col) != weight.size(Axis::col))
    {
        cerr << "Error: Data and weight matrices have different shape." << endl;
        throw runtime_error("Data and weight matrices have different shape.");
    }

    double              ret = 0;
    map<string, size_t> statistic;
    for (size_t r = 0; r < data.size(Axis::row); ++r)
    {
        for (size_t c = 0; c < data.size(Axis::col); ++c)
            if (0 == statistic.count(data.iloc(r, c)))
                statistic.insert({data.iloc(r, c), weight.iloc(r, c)});
            else
                statistic.at(data.iloc(r, c)) += weight.iloc(r, c);
    }
    for (const auto e : statistic)
    {
        double proportion = e.second / data.size();
        if (proportion > 0) ret -= (proportion * log2(proportion));
    }

    return ret;
}
} // namespace TL

#endif // ENTROPY_HPP