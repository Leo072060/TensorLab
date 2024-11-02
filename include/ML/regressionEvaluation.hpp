#ifndef REGRESSION_EVALUATION_HPP
#define REGRESSION_EVALUATION_HPP

#include "_internal/managed.hpp"
#include "mat/mat.hpp"

namespace TL
{
using namespace _internal;

template <typename T = double> class RegressionEvaluation : public ManagedClass
{
  public:
    RegressionEvaluation();
    RegressionEvaluation(const RegressionEvaluation &other);
    RegressionEvaluation(RegressionEvaluation &&other) noexcept;
    RegressionEvaluation<T> &operator=(const RegressionEvaluation &rhs);
    RegressionEvaluation<T> &operator=(RegressionEvaluation &&rhs) noexcept;
    RegressionEvaluation(const Mat<T> &y_pred, const Mat<T> &y_target);

  public:
    void fit(const Mat<T> &y_pred, const Mat<T> &y_target)
    {
        const_cast<const RegressionEvaluation *>(this)->fit(y_pred, y_target);
    }
    void report() const;
    T    mean_absolute_error() const;
    T    mean_squared_error() const;
    T    root_mean_squared_error() const;
    T    mean_absolute_percentage_error() const;
    T    r2_score() const;

  private:
    void fit(const Mat<T> &y_pred, const Mat<T> &y_target) const;

  private:
    mutable ManagedVal<Mat<T>> managed_y_target;
    mutable ManagedVal<Mat<T>> managed_y_pred;
    mutable ManagedVal<Mat<T>> managed_y_target_minus_y_pred;
    mutable ManagedVal<Mat<T>> managed_y_target_minus_mean_y_target;
    mutable ManagedVal<T>      managed_MAE;
    mutable ManagedVal<T>      managed_MSE;
    mutable ManagedVal<T>      managed_RMSE;
    mutable ManagedVal<T>      managed_MAPE;
    mutable ManagedVal<T>      managed_R2;
};
template <typename T>
RegressionEvaluation<T>::RegressionEvaluation()
    : ManagedClass()
    , managed_y_target(this->administrator)
    , managed_y_pred(this->administrator)
    , managed_y_target_minus_y_pred(this->administrator)
    , managed_y_target_minus_mean_y_target(this->administrator)
    , managed_MAE(this->administrator)
    , managed_MSE(this->administrator)
    , managed_RMSE(this->administrator)
    , managed_MAPE(this->administrator)
    , managed_R2(this->administrator)
{
}
template <typename T>
RegressionEvaluation<T>::RegressionEvaluation(const RegressionEvaluation &other)
    : ManagedClass(other)
    , managed_y_target(this->administrator)
    , managed_y_pred(this->administrator)
    , managed_y_target_minus_y_pred(this->administrator)
    , managed_y_target_minus_mean_y_target(this->administrator)
    , managed_MAE(this->administrator)
    , managed_MSE(this->administrator)
    , managed_RMSE(this->administrator)
    , managed_MAPE(this->administrator)
    , managed_R2(this->administrator)
{
}
template <typename T>
RegressionEvaluation<T>::RegressionEvaluation(RegressionEvaluation &&other) noexcept
    : ManagedClass(std::move(other))
    , managed_y_target(this->administrator)
    , managed_y_pred(this->administrator)
    , managed_y_target_minus_y_pred(this->administrator)
    , managed_y_target_minus_mean_y_target(this->administrator)
    , managed_MAE(this->administrator)
    , managed_MSE(this->administrator)
    , managed_RMSE(this->administrator)
    , managed_MAPE(this->administrator)
    , managed_R2(this->administrator)
{
    this->copyAfterConstructor(other);
}
template <typename T> RegressionEvaluation<T> &RegressionEvaluation<T>::operator=(const RegressionEvaluation &rhs)
{
    ManagedClass::operator=(rhs);
}
template <typename T> RegressionEvaluation<T> &RegressionEvaluation<T>::operator=(RegressionEvaluation &&rhs) noexcept
{
    using namespace std;
    ManagedClass::operator=(move(rhs));
}
template <typename T>
RegressionEvaluation<T>::RegressionEvaluation(const Mat<T> &y_pred, const Mat<T> &y_target)
    : ManagedClass()
    , managed_y_target(this->administrator)
    , managed_y_pred(this->administrator)
    , managed_y_target_minus_y_pred(this->administrator)
    , managed_y_target_minus_mean_y_target(this->administrator)
    , managed_MAE(this->administrator)
    , managed_MSE(this->administrator)
    , managed_RMSE(this->administrator)
    , managed_MAPE(this->administrator)
    , managed_R2(this->administrator)
{
    fit(y_pred, y_target);
}
template <typename T> void RegressionEvaluation<T>::report() const
{
    using namespace std;

    if (!managed_y_target.isReadable())
    {
        cerr << "Error: The model must be fitted before generating the report." << endl;
        throw runtime_error("Model not fitted.");
    }

    cout << "\t------- Regression Model Performance Report -------\n";
    cout << "mean absolute error           "
         << "\t" << mean_absolute_error() << endl;
    cout << "mean squared error            "
         << "\t" << mean_squared_error() << endl;
    cout << "root mean squared error       "
         << "\t" << root_mean_squared_error() << endl;
    cout << "mean absolute percentage error"
         << "\t" << mean_absolute_percentage_error() << endl;
    cout << "r2 score                      "
         << "\t" << r2_score() << endl;
}
template <typename T> T RegressionEvaluation<T>::mean_absolute_error() const
{
    using namespace std;

    if (managed_MAE.isReadable()) return managed_MAE.read();

    if (!managed_y_target.isReadable())
    {
        cerr << "Error: The model must be fitted before generating mean absolute error." << endl;
        throw runtime_error("Model not fitted.");
    }

    T ret = managed_y_target_minus_y_pred.read().abs().mean(Axis::all).iloc(0, 0);

    this->record(managed_MAE, ret);
    return ret;
}
template <typename T> T RegressionEvaluation<T>::mean_squared_error() const
{
    using namespace std;

    if (managed_MSE.isReadable()) return managed_MSE.read();

    if (!managed_y_target.isReadable())
    {
        cerr << "Error: The model must be fitted before generating mean squared error." << endl;
        throw runtime_error("Model not fitted.");
    }

    T ret = (managed_y_target_minus_y_pred.read().abs() ^ 0.5).mean(Axis::all).iloc(0, 0);

    this->record(managed_MSE, ret);
    return ret;
}
template <typename T> T RegressionEvaluation<T>::root_mean_squared_error() const
{
    using namespace std;

    if (managed_RMSE.isReadable()) return managed_RMSE.read();

    if (!managed_y_target.isReadable())
    {
        cerr << "Error: The model must be fitted before generating root mean squared error." << endl;
        throw runtime_error("Model not fitted.");
    }

    T ret = pow(mean_squared_error(), 0.5);

    this->record(managed_RMSE, ret);
    return ret;
}
template <typename T> T RegressionEvaluation<T>::mean_absolute_percentage_error() const
{
    using namespace std;

    if (managed_MAPE.isReadable()) return managed_MAPE.read();

    if (!managed_y_target.isReadable())
    {
        cerr << "Error: The model must be fitted before generating mean absolute percentage error." << endl;
        throw runtime_error("Model not fitted.");
    }

    T ret = (managed_y_target_minus_y_pred.read() / managed_y_target.read()).abs().mean(Axis::all).iloc(0, 0);

    this->record(managed_MAPE, ret);
    return ret;
}
template <typename T> T RegressionEvaluation<T>::r2_score() const
{
    using namespace std;

    if (managed_R2.isReadable()) return managed_R2.read();

    if (!managed_y_target.isReadable())
    {
        cerr << "Error: The model must be fitted before generating r2 score." << endl;
        throw runtime_error("Model not fitted.");
    }

    T ret = 1 - ((managed_y_target_minus_y_pred.read() ^ 2).sum(Axis::all) /
                 (managed_y_target_minus_mean_y_target.read() ^ 2).sum(Axis::all))
                    .iloc(0, 0);

    this->record(managed_R2, ret);
    return ret;
}
template <typename T> void RegressionEvaluation<T>::fit(const Mat<T> &y_pred, const Mat<T> &y_target) const
{
    using namespace std;

    if (y_pred.size(Axis::row) != y_target.size(Axis::row))
    {
        cerr << "Error: The number of rows in predicted values and target values must be the same." << endl;
        throw runtime_error("Dimension mismatch.");
    }
    if (y_pred.size(Axis::row) < 1)
    {
        cerr << "Error: The input matrices must have at least one row." << endl;
        throw invalid_argument("The input matrices must have at least one row.");
    }

    this->refresh();
    this->record(managed_y_target, y_target);
    this->record(managed_y_pred, y_pred);

    // calculate
    Mat<T> tmp(y_target.size(Axis::row), 1);
    // calculate managed_y_target_minus_y_pred
    for (size_t i = 0; i < y_target.size(Axis::row); ++i)
        tmp.iloc(i, 0) = y_target.iloc(i, 0) - y_pred.iloc(i, 0);
    this->record(managed_y_target_minus_y_pred, tmp);
    // calculate managed_y_target_minus_mean_y_target
    for (size_t i = 0; i < y_target.size(Axis::row); ++i)
        tmp.iloc(i, 0) = y_target.iloc(i, 0) - managed_y_target.read().mean(Axis::all).iloc(0, 0);
    this->record(managed_y_target_minus_mean_y_target, tmp);
}
} // namespace TL
#endif // REGRESSION_EVALUATION_HPP