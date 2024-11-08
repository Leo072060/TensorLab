#include"ML/evaluation_classification.hpp"

using namespace TL;

Evaluation_classification::Evaluation_classification()
    : ManagedClass()
    , managed_y_target(this->administrator)
    , managed_y_pred(this->administrator)
    , managed_confusionMatrix(this->administrator)
    , managed_accuracy(this->administrator)
    , managed_errorRate(this->administrator)
    , managed_percision(this->administrator)
    , managed_recall(this->administrator)
{
}
Evaluation_classification::Evaluation_classification(const Evaluation_classification &other)
    : ManagedClass(other)
    , managed_y_target(this->administrator, other.administrator, other.managed_y_target)
    , managed_y_pred(this->administrator, other.administrator, other.managed_y_pred)
    , managed_confusionMatrix(this->administrator, other.administrator, other.managed_confusionMatrix)
    , managed_accuracy(this->administrator, other.administrator, other.managed_accuracy)
    , managed_errorRate(this->administrator, other.administrator, other.managed_errorRate)
    , managed_percision(this->administrator, other.administrator, other.managed_percision)
    , managed_recall(this->administrator, other.administrator, other.managed_recall)
{
}
Evaluation_classification::Evaluation_classification(const Evaluation_classification &&other)
    : ManagedClass(other)
    , managed_y_target(this->administrator, other.administrator, other.managed_y_target)
    , managed_y_pred(this->administrator, other.administrator, other.managed_y_pred)
    , managed_confusionMatrix(this->administrator, other.administrator, other.managed_confusionMatrix)
    , managed_accuracy(this->administrator, other.administrator, other.managed_accuracy)
    , managed_errorRate(this->administrator, other.administrator, other.managed_errorRate)
    , managed_percision(this->administrator, other.administrator, other.managed_percision)
    , managed_recall(this->administrator, other.administrator, other.managed_recall)
{
}
Evaluation_classification &Evaluation_classification::operator=(const Evaluation_classification &rhs)
{
    ManagedClass::operator=(rhs);
    managed_y_target.copy(this->administrator, rhs.administrator, rhs.managed_y_target);
    managed_y_pred.copy(this->administrator, rhs.administrator, rhs.managed_y_pred);
    managed_confusionMatrix.copy(this->administrator, rhs.administrator, rhs.managed_confusionMatrix);
    managed_accuracy.copy(this->administrator, rhs.administrator, rhs.managed_accuracy);
    managed_errorRate.copy(this->administrator, rhs.administrator, rhs.managed_errorRate);
    managed_percision.copy(this->administrator, rhs.administrator, rhs.managed_percision);
    managed_recall.copy(this->administrator, rhs.administrator, rhs.managed_recall);
    return *this;
}
Evaluation_classification &Evaluation_classification::operator=(Evaluation_classification &&rhs) noexcept
{
    ManagedClass::operator=(rhs);
    managed_y_target.copy(this->administrator, rhs.administrator, rhs.managed_y_target);
    managed_y_pred.copy(this->administrator, rhs.administrator, rhs.managed_y_pred);
    managed_confusionMatrix.copy(this->administrator, rhs.administrator, rhs.managed_confusionMatrix);
    managed_accuracy.copy(this->administrator, rhs.administrator, rhs.managed_accuracy);
    managed_errorRate.copy(this->administrator, rhs.administrator, rhs.managed_errorRate);
    managed_percision.copy(this->administrator, rhs.administrator, rhs.managed_percision);
    managed_recall.copy(this->administrator, rhs.administrator, rhs.managed_recall);
    return *this;
}
void Evaluation_classification::fit(const Mat<std::string> &y_pred, const Mat<std::string> &y_target)
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
void Evaluation_classification::report() const
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
Mat<size_t> Evaluation_classification::confusionMatrix() const
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
    Mat<size_t> ret(types_y_target.size(Axis::col), types_y_target.size(Axis::col));
    for (size_t i = 0; i < managed_y_target.read().size(Axis::row); ++i)
    {
        ret.iloc(types_y_target.find(managed_y_target.read().iloc(i, 0)).cbegin()->second,
                 types_y_target.find(managed_y_pred.read().iloc(i, 0)).cbegin()->second) += 1;
    }
    for (size_t i = 0; i < ret.size(Axis::row); ++i)
    {
        ret.iloc_name(i, Axis::row) = types_y_target.iloc(0, i);
        ret.iloc_name(i, Axis::col) = types_y_target.iloc(0, i);
    }

    this->record(managed_confusionMatrix, ret);
    return ret;
}
double Evaluation_classification::accuracy() const
{
    using namespace std;

    if (!managed_y_target.isReadable())
    {
        cerr << "Error: The model must be fitted before generating accuracy." << endl;
        throw runtime_error("Model not fitted.");
    }

    if (managed_accuracy.isReadable()) return managed_accuracy.read();

    double sum_TFPN  = managed_confusionMatrix.read().sum();
    double sum_TP_TN = 0;
    for (size_t i = 0; i < managed_confusionMatrix.read().size(Axis::row); ++i)
        sum_TP_TN += managed_confusionMatrix.read().iloc(i, i);

    double ret = sum_TP_TN / sum_TFPN;

    this->record(managed_accuracy, ret);
    return ret;
}
double Evaluation_classification::error_rate() const
{
    using namespace std;

    if (managed_errorRate.isReadable()) return managed_errorRate.read();

    if (!managed_y_target.isReadable())
    {
        cerr << "Error: The model must be fitted before generating error rate." << endl;
        throw runtime_error("Model not fitted.");
    }

    double ret = 1 - accuracy();

    this->record(managed_errorRate, ret);
    return ret;
}
Mat<double> Evaluation_classification::percision() const
{
    using namespace std;

    if (managed_percision.isReadable()) return managed_percision.read();

    if (!managed_y_target.isReadable())
    {
        cerr << "Error: The model must be fitted before generating percision." << endl;
        throw runtime_error("Model not fitted.");
    }

    Mat<size_t> sum_TP_FP = managed_confusionMatrix.read().sum(Axis::row);
    Mat<double> ret(1, managed_confusionMatrix.read().size(Axis::col));
    for (size_t i = 0; i < ret.size(Axis::col); ++i)
    {
        ret.iloc(0, i)              = managed_confusionMatrix.read().iloc(i, i) * 1.0 / sum_TP_FP.iloc(i, 0);
        ret.iloc_name(i, Axis::col) = managed_confusionMatrix.read().iloc_name(i, Axis::col);
    }
    ret.iloc_name(0, Axis::row) = "percision";

    this->record(managed_percision, ret);
    return ret;
}
Mat<double> Evaluation_classification::recall() const
{
    using namespace std;

    if (managed_recall.isReadable()) return managed_recall.read();

    if (!managed_y_target.isReadable())
    {
        cerr << "Error: The model must be fitted before generating recall." << endl;
        throw runtime_error("Model not fitted.");
    }

    Mat<size_t> sum_TP_FP = managed_confusionMatrix.read().sum(Axis::col);
    Mat<double> ret(1, managed_confusionMatrix.read().size(Axis::col));
    for (size_t i = 0; i < ret.size(Axis::col); ++i)
    {
        ret.iloc(0, i)              = managed_confusionMatrix.read().iloc(i, i) * 1.0 / sum_TP_FP.iloc(0, i);
        ret.iloc_name(i, Axis::col) = managed_confusionMatrix.read().iloc_name(i, Axis::col);
    }
    ret.iloc_name(0, Axis::row) = "recall";

    this->record(managed_recall, ret);
    return ret;
}