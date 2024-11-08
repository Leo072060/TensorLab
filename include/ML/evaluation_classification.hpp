#ifndef EVALUATION_CLASSIFICATION_HPP
#define EVALUATION_CLASSIFICATION_HPP

#include "_internal/managed.hpp"
#include "mat/mat.hpp"

namespace TL
{
using namespace _internal;

class Evaluation_classification : public ManagedClass
{
  public:
    Evaluation_classification();
    Evaluation_classification(const Evaluation_classification &other);
    Evaluation_classification(const Evaluation_classification &&other);
    Evaluation_classification &operator=(const Evaluation_classification &rhs);
    Evaluation_classification &operator=(Evaluation_classification &&rhs) noexcept;

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

#endif // EVALUATION_CLASSIFICATION_HPP