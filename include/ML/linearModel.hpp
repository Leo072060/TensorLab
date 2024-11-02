#ifndef LINEAR_MODEL_HPP
#define LINEAR_MODEL_HPP

#include "ML/modelBase.hpp"
#include "mat/mat.hpp"

namespace TL
{
#pragma region LinearRegression
template <class T = double> class LinearRegression : public RegressionModelBase<T>
{
  private:
    Mat<T> train_(const Mat<T> &x, const Mat<T> &y) override;
    Mat<T> predict_(const Mat<T> &x, const Mat<T> &thetas) const override;

  public:
    std::shared_ptr<RegressionModelBase<T>> clone() const override;

  public:
    // model parameters
    double learning_rate = 0.0003;
    size_t batch_size    = 100;
    size_t iterations    = 1700;
};

template <typename T> Mat<T> LinearRegression<T>::train_(const Mat<T> &x, const Mat<T> &y)
{
    using namespace std;

    Mat<T> ones(x.size(Axis::row), 1);
    ones     = 1;
    Mat<T> w = x.concat(ones, Axis::col);
    Mat<T> thetas(1, x.size(Axis::col) + 1);

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

        Mat<T> tmp_thetas(thetas);

        for (size_t i = 0; i < w.size(Axis::col); ++i)
        {
            T tmp_theta_i = 0;
            for (auto &e : randomNums)
            {
                tmp_theta_i +=
                    learning_rate *
                    ((y.iloc(e, Axis::row) - thetas.dot(w.iloc(e, Axis::row).transpose())) * w.iloc(e, i)).iloc(0, 0);
            }
            tmp_thetas.iloc(0, i) += (tmp_theta_i / batch_size);
        }
        thetas = tmp_thetas;
    }
    return thetas;
}
template <typename T> Mat<T> LinearRegression<T>::predict_(const Mat<T> &x, const Mat<T> &thetas) const
{
    using namespace std;

    if (x.size(Axis::col) + 1 != thetas.size(Axis::col))
    {
        cerr << "Error: The input matrix has incompatible dimensions with the model parameters." << endl;
        cerr << "Expected columns: " << (thetas.size(Axis::col) - 1) << ", but got: " << x.size(Axis::col) << "."
             << endl;
        throw invalid_argument("The input matrix has incompatible dimensions with the model parameters.");
    }
    Mat<T> ones(x.size(Axis::row), 1);
    ones     = 1;
    Mat<T> w = x.concat(ones, Axis::col);
    return w.dot(thetas.transpose());
}
template <typename T> std::shared_ptr<RegressionModelBase<T>> LinearRegression<T>::clone() const
{
    using namespace std;

    return make_shared<LinearRegression<T>>(*this);
}
#pragma endregion

#pragma region LogisticRegression
template <typename T = double> class LogisticRegression : public BinaryClassificationModelBase<T>
{
  private:
    Mat<T>           train_binary(const Mat<T> &x, const Mat<std::string> &y) override;
    Mat<std::string> predict_binary(const Mat<T> &x, const Mat<T> &theta) const override;
    static Mat<T>    predict_probabilities(const Mat<T> &x, const Mat<T> &thetas);

  public:
    std::shared_ptr<ClassificationModelBase<T>> clone() const override;

  public:
    // model parameters
    double learning_rate = 0.0003;
    size_t batch_size    = 50;
    size_t iterations    = 1700;
};

template <typename T> Mat<T> LogisticRegression<T>::train_binary(const Mat<T> &x, const Mat<std::string> &y)
{
    using namespace std;

    Mat<string> labels = y.unique();
    if (labels.size() != 2)
    {
        cerr << "Error: The target variable must have exactly 2 unique labels for binary classification." << endl;
        throw invalid_argument("Invalid number of labels for binary classification.");
    }

    Mat<T> mumerical_y(y.size(Axis::row), y.size(Axis::col));
    for (size_t i = 0; i < y.size(Axis::row); ++i)
        mumerical_y.iloc(i, 0) = (labels.iloc(0, 0) == y.iloc(i, 0) ? 1 : 0);

    Mat<T> ones(x.size(Axis::row), 1);
    ones     = 1;
    Mat<T> w = x.concat(ones, Axis::col);
    Mat<T> thetas(1, x.size(Axis::col) + 1);

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
        uniform_int_distribution<> dis(0, x.size_row() - 1);
        while (randomNums.size() < batch_size)
            randomNums.insert(dis(gen));

        Mat<T> tmp_thetas(thetas);

        for (size_t i = 0; i < w.size(Axis::col); ++i)
        {
            T tmp_theta_i = 0;
            // gradient descent
            for (const auto &e : randomNums)
            {
                tmp_theta_i +=
                    learning_rate * ((mumerical_y.iloc(e, Axis::row) -
                                      LogisticRegression<T>::predict_probabilities(x.iloc(e, Axis::row), thetas)) *
                                     w.iloc(e, i))
                                        .iloc(0, 0);
            }
            tmp_thetas.iloc(0, i) += (tmp_theta_i);
        }
        thetas = tmp_thetas;
        display_rainbow(thetas);
    }

    return thetas;
}
template <typename T> Mat<std::string> LogisticRegression<T>::predict_binary(const Mat<T> &x, const Mat<T> &theta) const
{
    using namespace std;

    Mat<T>      probabilities = predict_probabilities(x, this->thetas);
    Mat<string> ret(x.size(Axis::row), 1);
    for (size_t i = 0; i < x.size(Axis::row); ++i)
        ret.iloc(i, 0) = probabilities.iloc(i, 0) > 0.5 ? this->managed_labels.read().iloc(0, 0)
                                                        : this->managed_labels.read().iloc(0, 1);
    return ret;
}
template <typename T> Mat<T> LogisticRegression<T>::predict_probabilities(const Mat<T> &x, const Mat<T> &thetas)
{
    using namespace std;

    if (x.size(Axis::col) + 1 != thetas.size(Axis::col))
        throw invalid_argument("Error: Number of columns in x must be equal to number of columns in thetas minus one.");

    Mat<T> y(x.size(Axis::row), 1);
    Mat<T> ones(x.size(Axis::row), 1);
    ones     = 1;
    Mat<T> w = x.concat(ones, Axis::col);
    w.dot(thetas.transpose());
    for (size_t i = 0; i < x.size(Axis::row); ++i)
        y.iloc(i, 0) = 1 / (1 + exp(-w.iloc(i, 0)));

    return y;
}
template <typename T> std::shared_ptr<ClassificationModelBase<T>> LogisticRegression<T>::clone() const
{
    using namespace std;

    return make_shared<ClassificationModelBase<T>>(*this);
}
#pragma endregion
} // namespace TL
#endif // LINEAR_MODEL_H