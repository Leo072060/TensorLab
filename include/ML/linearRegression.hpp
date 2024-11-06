#ifndef LINEAR_REGRESSION_HPP
#define LINEAR_REGRESSION_HPP

#include "ML/_internal/regressionModelBase.hpp" 

namespace TL
{
using namespace _internal;

template <class T = double> class LinearRegression : public RegressionModelBase<T>
{
    // hook functions
  private:
    Mat<double> train_(const Mat<T> &x, const Mat<T> &y) override;
    Mat<T> predict_(const Mat<T> &x, const Mat<double> &theta) const override;

    // for polymorphism
  public:
    std::shared_ptr<RegressionModelBase<T>> clone() const override;

  public:
    // model parameters
    double learning_rate = 0.0003;
    size_t batch_size    = 100;
    size_t iterations    = 1700;
};

// hook functions
template <typename T> Mat<double> LinearRegression<T>::train_(const Mat<T> &x, const Mat<T> &y)
{
    using namespace std;

    Mat<T> ones(x.size(Axis::row), 1);
    ones     = 1;
    Mat<T> w = x.concat(ones, Axis::col);
    Mat<double> theta(1, x.size(Axis::col) + 1);

    // start training
    for (size_t I = 0; I < iterations; ++I)
    {
        if (x.size(Axis::row) < batch_size)
        {
            cerr << "Error: Batch size (" << batch_size << ") is larger than the available rows (" << x.size(Axis::row)
                 << ")." << endl;
            throw out_of_range("Batch size is larger than the available rows.");
        }

        // generate random numbers
        set<size_t>                randomNums;
        random_device              rd;
        mt19937                    gen(rd());
        uniform_int_distribution<> dis(0, x.size(Axis::row) - 1);
        while (randomNums.size() < batch_size)
            randomNums.insert(dis(gen));

        Mat<double> tmp_theta(theta);

        for (size_t i = 0; i < w.size(Axis::col); ++i)
        {
            T tmp_theta_i = 0;
            for (auto &e : randomNums)
            {
                tmp_theta_i +=
                    learning_rate *
                    ((y.iloc(e, Axis::row) - theta.dot(w.iloc(e, Axis::row).transpose())) * w.iloc(e, i)).iloc(0, 0);
            }
            tmp_theta.iloc(0, i) += (tmp_theta_i / batch_size);
        }
        theta = tmp_theta;
    }
    return theta;
}
template <typename T> Mat<T> LinearRegression<T>::predict_(const Mat<T> &x, const Mat<double> &theta) const
{
    using namespace std;

    if (x.size(Axis::col) + 1 != theta.size(Axis::col))
    {
        cerr << "Error: The input matrix has incompatible dimensions with the model parameters." << endl;
        cerr << "Expected columns: " << (theta.size(Axis::col) - 1) << ", but got: " << x.size(Axis::col) << "."
             << endl;
        throw invalid_argument("The input matrix has incompatible dimensions with the model parameters.");
    }
    Mat<T> ones(x.size(Axis::row), 1);
    ones     = 1;
    Mat<T> w = x.concat(ones, Axis::col);
    return w.dot(theta.transpose());
}

// for polymorphism
template <typename T> std::shared_ptr<RegressionModelBase<T>> LinearRegression<T>::clone() const
{
    using namespace std;

    return make_shared<LinearRegression<T>>(*this);
}
} // namespace TL
#endif // LINEAR_REGRESSION_HPP