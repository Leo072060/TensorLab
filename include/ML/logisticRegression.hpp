#ifndef LOGISTIC_REGRESSION_HPP
#define LOGISTIC_REGRESSION_HPP

#include "ML/_internal/binaryClassificationModelBase.hpp"

namespace TL
{
using namespace _internal;

class LogisticRegression : public BinaryClassificationModelBase<double>
{
  private:
    Mat<double>        train_binary(const Mat<double> &x, const Mat<std::string> &y) override;
    Mat<std::string>   predict_binary(const Mat<double> &x, const Mat<double> &theta) const override;
    static Mat<double> predict_probabilities(const Mat<double> &x, const Mat<double> &thetas);

  public:
    std::shared_ptr<ClassificationModelBase<double>> clone() const override;

  public:
    // model parameters
    double learning_rate = 0.00025;
    size_t batch_size    = 10;
    size_t iterations    = 300;
};
} // namespace TL
#endif // LOGISTIC_REGRESSION_HPP