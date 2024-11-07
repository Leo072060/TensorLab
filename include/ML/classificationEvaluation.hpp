#ifndef CLASSIFICATION_EVALUATION_HPP
#define CLASSIFICATION_EVALUATION_HPP

#include "_internal/managed.hpp"
#include "mat/mat.hpp"

namespace TL
{
using namespace _internal;

class ClassificationEvaluation : public ManagedClass
{
  public:
    ClassificationEvaluation();
    ClassificationEvaluation(const ClassificationEvaluation &other);
    ClassificationEvaluation(const ClassificationEvaluation &&other);
    ClassificationEvaluation &operator=(const ClassificationEvaluation &rhs);
    ClassificationEvaluation &operator=(ClassificationEvaluation &&rhs) noexcept;

  public:
    void        fit(const Mat<std::string> &y_pred, const Mat<std::string> &y_target);
    void        report() const;
    Mat<size_t> confusionMatrix() const;
    double      accuracy() const;
    double      error_rate() const;
    Mat<double> percision() const;
    Mat<double> recall() const;

  private:
    mutable ManagedVal<Mat<std::string>> managed_y_target;
    mutable ManagedVal<Mat<std::string>> managed_y_pred;
    mutable ManagedVal<Mat<size_t>>      managed_confusionMatrix;
    mutable ManagedVal<double>           managed_accuracy;
    mutable ManagedVal<double>           managed_errorRate;
    mutable ManagedVal<Mat<double>>      managed_percision;
    mutable ManagedVal<Mat<double>>      managed_recall;
};
} // namespace TL

#endif // CLASSIFICATION_EVALUATION_HPP