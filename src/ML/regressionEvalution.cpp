#include "ML/regressionEvaluation.hpp"

using namespace TL;

RegressionEvaluation ::RegressionEvaluation()
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
RegressionEvaluation::RegressionEvaluation(const RegressionEvaluation &other)
    : ManagedClass(other)
    , managed_y_target(this->administrator, other.administrator, other.managed_y_target)
    , managed_y_pred(this->administrator, other.administrator, other.managed_y_pred)
    , managed_y_target_minus_y_pred(this->administrator, other.administrator, other.managed_y_target_minus_y_pred)
    , managed_y_target_minus_mean_y_target(this->administrator, other.administrator,
                                           other.managed_y_target_minus_mean_y_target)
    , managed_MAE(this->administrator, other.administrator, other.managed_MAE)
    , managed_MSE(this->administrator, other.administrator, other.managed_MSE)
    , managed_RMSE(this->administrator, other.administrator, other.managed_RMSE)
    , managed_MAPE(this->administrator, other.administrator, other.managed_MAPE)
    , managed_R2(this->administrator, other.administrator, other.managed_R2)
{
}
RegressionEvaluation ::RegressionEvaluation(RegressionEvaluation &&other) noexcept
    : ManagedClass(std::move(other))
    , managed_y_target(this->administrator, other.administrator, other.managed_y_target)
    , managed_y_pred(this->administrator, other.administrator, other.managed_y_pred)
    , managed_y_target_minus_y_pred(this->administrator, other.administrator, other.managed_y_target_minus_y_pred)
    , managed_y_target_minus_mean_y_target(this->administrator, other.administrator,
                                           other.managed_y_target_minus_mean_y_target)
    , managed_MAE(this->administrator, other.administrator, other.managed_MAE)
    , managed_MSE(this->administrator, other.administrator, other.managed_MSE)
    , managed_RMSE(this->administrator, other.administrator, other.managed_RMSE)
    , managed_MAPE(this->administrator, other.administrator, other.managed_MAPE)
    , managed_R2(this->administrator, other.administrator, other.managed_R2)
{
}
RegressionEvaluation &RegressionEvaluation ::operator=(const RegressionEvaluation &rhs)
{
    ManagedClass::operator=(rhs);
    managed_y_target.copy(this->administrator, rhs.administrator, rhs.managed_y_target);
    managed_y_pred.copy(this->administrator, rhs.administrator, rhs.managed_y_pred);
    managed_y_target_minus_y_pred.copy(this->administrator, rhs.administrator, rhs.managed_y_target_minus_y_pred);
    managed_y_target_minus_mean_y_target.copy(this->administrator, rhs.administrator,
                                              rhs.managed_y_target_minus_mean_y_target);
    managed_MAE.copy(this->administrator, rhs.administrator, rhs.managed_MAE);
    managed_MSE.copy(this->administrator, rhs.administrator, rhs.managed_MSE);
    managed_RMSE.copy(this->administrator, rhs.administrator, rhs.managed_RMSE);
    managed_MAPE.copy(this->administrator, rhs.administrator, rhs.managed_MAPE);
    managed_R2.copy(this->administrator, rhs.administrator, rhs.managed_R2);
    return *this;
}
RegressionEvaluation &RegressionEvaluation ::operator=(RegressionEvaluation &&rhs) noexcept
{
    using namespace std;
    ManagedClass::operator=(move(rhs));
    managed_y_target.copy(this->administrator, rhs.administrator, rhs.managed_y_target);
    managed_y_pred.copy(this->administrator, rhs.administrator, rhs.managed_y_pred);
    managed_y_target_minus_y_pred.copy(this->administrator, rhs.administrator, rhs.managed_y_target_minus_y_pred);
    managed_y_target_minus_mean_y_target.copy(this->administrator, rhs.administrator,
                                              rhs.managed_y_target_minus_mean_y_target);
    managed_MAE.copy(this->administrator, rhs.administrator, rhs.managed_MAE);
    managed_MSE.copy(this->administrator, rhs.administrator, rhs.managed_MSE);
    managed_RMSE.copy(this->administrator, rhs.administrator, rhs.managed_RMSE);
    managed_MAPE.copy(this->administrator, rhs.administrator, rhs.managed_MAPE);
    managed_R2.copy(this->administrator, rhs.administrator, rhs.managed_R2);
    return *this;
}
void RegressionEvaluation ::fit(const Mat<double> &y_pred, const Mat<double> &y_target)
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
    Mat<double> tmp(y_target.size(Axis::row), 1);
    // calculate managed_y_target_minus_y_pred
    for (size_t i = 0; i < y_target.size(Axis::row); ++i)
        tmp.iloc(i, 0) = y_target.iloc(i, 0) - y_pred.iloc(i, 0);
    this->record(managed_y_target_minus_y_pred, tmp);
    // calculate managed_y_target_minus_mean_y_target
    for (size_t i = 0; i < y_target.size(Axis::row); ++i)
        tmp.iloc(i, 0) = y_target.iloc(i, 0) - managed_y_target.read().mean(Axis::all).iloc(0, 0);
    this->record(managed_y_target_minus_mean_y_target, tmp);
}
void RegressionEvaluation ::report() const
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
double RegressionEvaluation ::mean_absolute_error() const
{
    using namespace std;

    if (managed_MAE.isReadable()) return managed_MAE.read();

    if (!managed_y_target.isReadable())
    {
        cerr << "Error: The model must be fitted before generating mean absolute error." << endl;
        throw runtime_error("Model not fitted.");
    }

    double ret = managed_y_target_minus_y_pred.read().abs().mean(Axis::all).iloc(0, 0);

    this->record(managed_MAE, ret);
    return ret;
}
double RegressionEvaluation ::mean_squared_error() const
{
    using namespace std;

    if (managed_MSE.isReadable()) return managed_MSE.read();

    if (!managed_y_target.isReadable())
    {
        cerr << "Error: The model must be fitted before generating mean squared error." << endl;
        throw runtime_error("Model not fitted.");
    }

    double ret = (managed_y_target_minus_y_pred.read().abs() ^ 0.5).mean(Axis::all).iloc(0, 0);

    this->record(managed_MSE, ret);
    return ret;
}
double RegressionEvaluation ::root_mean_squared_error() const
{
    using namespace std;

    if (managed_RMSE.isReadable()) return managed_RMSE.read();

    if (!managed_y_target.isReadable())
    {
        cerr << "Error: The model must be fitted before generating root mean squared error." << endl;
        throw runtime_error("Model not fitted.");
    }

    double ret = pow(mean_squared_error(), 0.5);

    this->record(managed_RMSE, ret);
    return ret;
}
double RegressionEvaluation ::mean_absolute_percentage_error() const
{
    using namespace std;

    if (managed_MAPE.isReadable()) return managed_MAPE.read();

    if (!managed_y_target.isReadable())
    {
        cerr << "Error: The model must be fitted before generating mean absolute percentage error." << endl;
        throw runtime_error("Model not fitted.");
    }

    double ret = (managed_y_target_minus_y_pred.read() / managed_y_target.read()).abs().mean(Axis::all).iloc(0, 0);

    this->record(managed_MAPE, ret);
    return ret;
}
double RegressionEvaluation ::r2_score() const
{
    using namespace std;

    if (managed_R2.isReadable()) return managed_R2.read();

    if (!managed_y_target.isReadable())
    {
        cerr << "Error: The model must be fitted before generating r2 score." << endl;
        throw runtime_error("Model not fitted.");
    }

    double ret = 1 - ((managed_y_target_minus_y_pred.read() ^ 2).sum(Axis::all) /
                      (managed_y_target_minus_mean_y_target.read() ^ 2).sum(Axis::all))
                         .iloc(0, 0);

    this->record(managed_R2, ret);
    return ret;
}