#ifndef LINEAR_MODEL_HPP
#define LINEAR_MODEL_HPP

#include "ML/modelBase.hpp"
#include "mat/mat.hpp"

#pragma region LinearRegression
template <class T = double> class LinearRegression : public RegressionModelBase<T>
{
  private:
    void   train_(const Mat<T> &x, const Mat<T> &y) override;
    Mat<T> predict_(const Mat<T> &x) const override;

  public:
    std::shared_ptr<RegressionModelBase<T>> clone() const override;

  public:
    // model parameters
    double learning_rate = 0.0003;
    size_t batch_size    = 100;
    size_t iterations    = 1700;
};

template <typename T> void LinearRegression<T>::train_(const Mat<T> &x, const Mat<T> &y)
{
    using namespace std;

    Mat<T> ones(x.size(Axis::row), 1);
    ones     = 1;
    Mat<T> w = x.concat(ones, Axis::col);
    Mat<T> thetas(1, x.size(Axis::col) + 1);

    // start training
    for (size_t i = 0; i < iterations; ++i)
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

        Mat<T> tmp_thetas(thetas);

        for (size_t i = 0; i < w.size(Axis::col); ++i)
        {
            T tmp_theta_i = 0;
            for (auto &e : randomNums)
            {
                tmp_theta_i +=
                    learning_rate *
                    ((y.iloc(e, Axis::row) - thetas.dot(w.iloc(e, Axis::row).transpose()) * w.iloc(e, i)).iloc(0, 0));
            }
            tmp_thetas.iloc(0, i) += (tmp_theta_i / batch_size);
        }
        thetas = tmp_thetas;
    }
    this->record(this->managed_thetas, thetas);
}
template <typename T> Mat<T> LinearRegression<T>::predict_(const Mat<T> &x) const
{
    using namespace std;

    if (x.size(Axis::col) + 1 != this->managed_thetas.read().size(Axis::col))
    {
        cerr << "Error: The input matrix has incompatible dimensions with the model parameters." << endl;
        cerr << "Expected columns: " << (this->managed_thetas.read().size(Axis::col) - 1)
             << ", but got: " << x.size(Axis::col) << "." << endl;
        throw invalid_argument("The input matrix has incompatible dimensions with the model parameters.");
    }
    Mat<T> ones(x.size(Axis::row), 1);
    ones     = 1;
    Mat<T> w = x.concat(ones, Axis::col);
    return w.dot(this->managed_thetas.read().transpose());
}
template <typename T> std::shared_ptr<RegressionModelBase<T>> LinearRegression<T>::clone() const
{
    using namespace std;

    return make_shared<LinearRegression<T>>(*this);
}
#pragma endregion

#endif // LINEAR_MODEL_H