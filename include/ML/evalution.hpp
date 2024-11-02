#include <cmath>
#include <iostream>
#include <stdexcept>

#include "_internal/managed.hpp"
#include "mat/mat.hpp"

namespace TL
{
using namespace _internal;

#pragma region RegressionEvaluation
template <typename T = double> class RegressionEvaluation : public ManagedClass
{
  public:
    RegressionEvaluation();
    RegressionEvaluation(const RegressionEvaluation &other);
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
    : ManagedClass(), managed_y_target(this->administrator), managed_y_pred(this->administrator),
      managed_y_target_minus_y_pred(this->administrator), managed_y_target_minus_mean_y_target(this->administrator),
      managed_MAE(this->administrator), managed_MSE(this->administrator), managed_RMSE(this->administrator),
      managed_MAPE(this->administrator), managed_R2(this->administrator)
{
}
template <typename T>
RegressionEvaluation<T>::RegressionEvaluation(const RegressionEvaluation &other)
    : ManagedClass(other), managed_y_target(this->administrator), managed_y_pred(this->administrator),
      managed_y_target_minus_y_pred(this->administrator), managed_y_target_minus_mean_y_target(this->administrator),
      managed_MAE(this->administrator), managed_MSE(this->administrator), managed_RMSE(this->administrator),
      managed_MAPE(this->administrator), managed_R2(this->administrator)
{
}
template <typename T> RegressionEvaluation<T> &RegressionEvaluation<T>::operator=(const RegressionEvaluation &rhs)
{
    ManagedClass::operator=(rhs);
}
template <typename T> RegressionEvaluation<T> &RegressionEvaluation<T>::operator=(RegressionEvaluation &&rhs) noexcept
{
    ManagedClass::operator=(rhs);
}
template <typename T>
RegressionEvaluation<T>::RegressionEvaluation(const Mat<T> &y_pred, const Mat<T> &y_target)
    : ManagedClass(), managed_y_target(this->administrator), managed_y_pred(this->administrator),
      managed_y_target_minus_y_pred(this->administrator), managed_y_target_minus_mean_y_target(this->administrator),
      managed_MAE(this->administrator), managed_MSE(this->administrator), managed_RMSE(this->administrator),
      managed_MAPE(this->administrator), managed_R2(this->administrator)
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

    this->record(managed_MAE, managed_y_target_minus_y_pred.read().abs().mean(Axis::all).iloc(0, 0));
    return managed_MAE;
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

    this->record(managed_MSE, (managed_y_target_minus_y_pred.read().abs() ^ 0.5).mean(Axis::all).iloc(0, 0));
    return managed_MSE;
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

    this->record(managed_RMSE, pow(mean_squared_error(), 0.5));
    return managed_RMSE;
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

    this->record(managed_MAPE,
                 (managed_y_target_minus_y_pred.read() / managed_y_target).abs().mean(Axis::all).iloc(0, 0));
    return managed_MAPE;
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

    this->record(managed_R2, 1 - ((managed_y_target_minus_y_pred.read() ^ 2).sum(Axis::all) /
                                  (managed_y_target_minus_mean_y_target.read() ^ 2).sum(Axis::all))
                                     .iloc(0, 0));
    return managed_R2;
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
#pragma endregion

#pragma region ClassificationEvaluation
template <typename T> class ClassificationEvaluation : public ManagedClass
{
  public:
    ClassificationEvaluation();
    ClassificationEvaluation(const ClassificationEvaluation<T> &other);
    ClassificationEvaluation(const ClassificationEvaluation<T> &&other);
    ClassificationEvaluation<T> &operator=(const ClassificationEvaluation<T> &rhs);
    ClassificationEvaluation<T> &operator=(ClassificationEvaluation<T> &&rhs) noexcept;
    ClassificationEvaluation(const Mat<std::string> &y_pred, const Mat<std::string> &y_target);

  public:
    void fit(const Mat<std::string> &y_pred, const Mat<std::string> &y_target)
    {
        const_cast<const ClassificationEvaluation<T> *>(this)->fit(y_pred, y_target);
    }
    void        report() const;
    Mat<size_t> confusionMatrix() const;
    T           accuracy() const;
    T           error_rate() const;
    Mat<T>      percision() const;
    Mat<T>      recall() const;

  private:
    void fit(const Mat<std::string> &y_pred, const Mat<std::string> &y_target) const;

  private:
    mutable ManagedVal<Mat<std::string>> managed_y_target;
    mutable ManagedVal<Mat<std::string>> managed_y_pred;
    mutable ManagedVal<Mat<size_t>>      managed_confusionMatrix;
    mutable ManagedVal<T>                managed_accuracy;
    mutable ManagedVal<T>                managed_errorRate;
    mutable ManagedVal<Mat<T>>           managed_percision;
    mutable ManagedVal<Mat<T>>           managed_recall;
};
template <typename T>
ClassificationEvaluation<T>::ClassificationEvaluation()
    : ManagedClass(), managed_y_target(this->administrator), managed_y_pred(this->administrator),
      managed_confusionMatrix(this->administrator), managed_accuracy(this->administrator),
      managed_errorRate(this->administrator), managed_percision(this->administrator),
      managed_recall(this->administrator)
{
}
template <typename T>
ClassificationEvaluation<T>::ClassificationEvaluation(const ClassificationEvaluation<T> &other)
    : ManagedClass(other), managed_y_target(this->administrator), managed_y_pred(this->administrator),
      managed_confusionMatrix(this->administrator), managed_accuracy(this->administrator),
      managed_errorRate(this->administrator), managed_percision(this->administrator),
      managed_recall(this->administrator)
{
    this->copyAfterConstructor(other);
}
template <typename T>
ClassificationEvaluation<T>::ClassificationEvaluation(const ClassificationEvaluation<T> &&other)
    : ManagedClass(other), managed_y_target(this->administrator), managed_y_pred(this->administrator),
      managed_confusionMatrix(this->administrator), managed_accuracy(this->administrator),
      managed_errorRate(this->administrator), managed_percision(this->administrator),
      managed_recall(this->administrator)
{
    this->copyAfterConstructor(other);
}
template <typename T>
ClassificationEvaluation<T> &ClassificationEvaluation<T>::operator=(const ClassificationEvaluation<T> &rhs)
{
    ManagedClass::operator=(rhs);
}
template <typename T>
ClassificationEvaluation<T> &ClassificationEvaluation<T>::operator=(ClassificationEvaluation<T> &&rhs) noexcept
{
    ManagedClass::operator=(rhs);
}
template <typename T>
ClassificationEvaluation<T>::ClassificationEvaluation(const Mat<std::string> &y_pred, const Mat<std::string> &y_target)
    : ClassificationEvaluation()
{
    fit(y_pred, y_target);
}
template <typename T> void ClassificationEvaluation<T>::report() const
{
    using namespace std;

    if (!managed_y_target.isReadable())
    {
        cerr << "Error: The model must be fitted before generating the report." << endl;
        throw runtime_error("Model not fitted.");
    }

    cout << "\t------- Classification Model Performance Report -------\n";
    cout << "confusion matrix : " << endl;
    display(managed_confusionMatrix.read(), both_names);
    cout << "accuracy                      "
         << "\t" << accuracy() << endl;
    cout << "error rate                    "
         << "\t" << error_rate() << endl;
    cout << "percision : " << endl;
    display(percision(), both_names);
    cout << "recall : " << endl;
    display(recall(), both_names);
}
template <typename T> Mat<size_t> ClassificationEvaluation<T>::confusionMatrix() const
{
    using namespace std;

    if (managed_confusionMatrix.isReadable()) return managed_confusionMatrix.read();

    if (!managed_y_target.isReadable())
    {
        cerr << "Error: The model must be fitted before generating confusion matrix." << endl;
        throw runtime_error("Model not fitted.");
    }

    Mat<string> types_y_target = managed_y_target.read().unique();
    Mat<string> types_y_pred   = managed_y_pred.read().unique();
    Mat<string> types_y        = types_y_target.concat(types_y_pred, Axis::col).unique();
    if (types_y.size() > types_y_target.size())
    {
        cerr << "Error: Predicted labels contain classes not present in the target labels." << endl;
        throw logic_error("Predicted labels contain classes not present in the target labels.");
    }
    types_y_target.sort(0, Order::asce, Axis::col);
    Mat<size_t> confusionMat(types_y_target.size(Axis::col), types_y_target.size(Axis::col));
    for (size_t i = 0; i < managed_y_target.read().size(Axis::row); ++i)
    {
        confusionMat.iloc(types_y_target.find(managed_y_target.read().iloc(i, 0)).cbegin()->second,
                          types_y_target.find(managed_y_pred.read().iloc(i, 0)).cbegin()->second) += 1;
    }
    for (size_t i = 0; i < confusionMat.size(Axis::row); ++i)
    {
        confusionMat.iloc_name(i, Axis::row) = types_y_target.iloc(0, i);
        confusionMat.iloc_name(i, Axis::col) = types_y_target.iloc(0, i);
    }

    this->record(managed_confusionMatrix, confusionMat);
    return confusionMat;
}
template <typename T> T ClassificationEvaluation<T>::accuracy() const
{
    using namespace std;

    if (!managed_y_target.isReadable())
    {
        cerr << "Error: The model must be fitted before generating accuracy." << endl;
        throw runtime_error("Model not fitted.");
    }

    if (managed_accuracy.isReadable()) return managed_accuracy.read();

    T sum_TFPN  = managed_confusionMatrix.read().sum();
    T sum_TP_TN = 0;
    for (size_t i = 0; i < managed_confusionMatrix.read().size(Axis::row); ++i)
        sum_TP_TN += managed_confusionMatrix.read().iloc(i, i);

    this->record(managed_accuracy, sum_TP_TN / sum_TFPN);
    return managed_accuracy.read();
}
template <typename T> T ClassificationEvaluation<T>::error_rate() const
{
    using namespace std;

    if (managed_errorRate.isReadable()) return managed_errorRate.read();

    if (!managed_y_target.isReadable())
    {
        cerr << "Error: The model must be fitted before generating error rate." << endl;
        throw runtime_error("Model not fitted.");
    }

    this->record(managed_errorRate, 1 - accuracy());
    return managed_errorRate;
}
template <typename T> Mat<T> ClassificationEvaluation<T>::percision() const
{
    using namespace std;

    if (managed_percision.isReadable()) return managed_percision.read();

    if (!managed_y_target.isReadable())
    {
        cerr << "Error: The model must be fitted before generating percision." << endl;
        throw runtime_error("Model not fitted.");
    }

    Mat<size_t> sum_TP_FP = managed_confusionMatrix.read().sum(Axis::row);
    Mat<T>      ret(1, managed_confusionMatrix.read().size(Axis::col));
    for (size_t i = 0; i < ret.size(Axis::col); ++i)
    {
        ret.iloc(0, i)              = managed_confusionMatrix.read().iloc(i, i) * 1.0 / sum_TP_FP.iloc(i, 0);
        ret.iloc_name(i, Axis::col) = managed_confusionMatrix.read().iloc_name(i, Axis::col);
    }
    ret.iloc_name(0, Axis::row) = "percision";

    this->record(managed_percision, ret);
    return managed_percision;
}
template <typename T> Mat<T> ClassificationEvaluation<T>::recall() const
{
    using namespace std;

    if (managed_recall.isReadable()) return managed_recall.read();

    if (!managed_y_target.isReadable())
    {
        cerr << "Error: The model must be fitted before generating recall." << endl;
        throw runtime_error("Model not fitted.");
    }

    Mat<size_t> sum_TP_FP = managed_confusionMatrix.read().sum(Axis::col);
    Mat<T>      ret(1, managed_confusionMatrix.read().size(Axis::col));
    for (size_t i = 0; i < ret.size(Axis::col); ++i)
    {
        ret.iloc(0, i)              = managed_confusionMatrix.read().iloc(i, i) * 1.0 / sum_TP_FP.iloc(0, i);
        ret.iloc_name(i, Axis::col) = managed_confusionMatrix.read().iloc_name(i, Axis::col);
    }
    ret.iloc_name(0, Axis::row) = "recall";

    this->record(managed_recall, ret);
    return managed_recall;
}
template <typename T>
void ClassificationEvaluation<T>::fit(const Mat<std::string> &y_pred, const Mat<std::string> &y_target) const
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
    Mat<string> types_y_pred   = y_pred.unique();
    Mat<string> types_y_target = y_target.unique();

    confusionMatrix();
}
} // namespace TL
#pragma endregion