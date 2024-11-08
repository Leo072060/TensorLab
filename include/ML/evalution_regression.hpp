#ifndef REGRESSION_EVALUATION_HPP
#define REGRESSION_EVALUATION_HPP

#include "_internal/managed.hpp"
#include "mat/mat.hpp"

namespace TL
{
using namespace _internal;

class Evaluation_regression : public ManagedClass
{
  public:
    Evaluation_regression();
    Evaluation_regression(const Evaluation_regression &other);
    Evaluation_regression(Evaluation_regression &&other) noexcept;
    Evaluation_regression  &operator=(const Evaluation_regression &rhs);
    Evaluation_regression  &operator=(Evaluation_regression &&rhs) noexcept;

  public:
    void fit(const Mat<double> &y_pred, const Mat<double> &y_target);
    void report() const;
    double    mean_absolute_error() const;
    double    mean_squared_error() const;
    double    root_mean_squared_error() const;
    double    mean_absolute_percentage_error() const;
    double    r2_score() const;

  private:
    mutable ManagedVal<Mat<double>> managed_y_target;
    mutable ManagedVal<Mat<double>> managed_y_pred;
    mutable ManagedVal<Mat<double>> managed_y_target_minus_y_pred;
    mutable ManagedVal<Mat<double>> managed_y_target_minus_mean_y_target;
    mutable ManagedVal<double>      managed_MAE;
    mutable ManagedVal<double>      managed_MSE;
    mutable ManagedVal<double>      managed_RMSE;
    mutable ManagedVal<double>      managed_MAPE;
    mutable ManagedVal<double>      managed_R2;
};
} // namespace TL
#endif // REGRESSION_EVALUATION_HPP