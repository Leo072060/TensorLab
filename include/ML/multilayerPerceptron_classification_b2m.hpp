#ifndef MULTILAYER_PERCEPTRON_CLASSIFICATION_B2M_HPP
#define MULTILAYER_PERCEPTRON_CLASSIFICATION_B2M_HPP

#include "ML/_internal/binaryClassificationModelBase.hpp"

namespace TL
{
using namespace _internal;

class MultilayerPerception_classification_b2m : public BinaryClassificationModelBase<double>
{
  private:
    Mat<double>        train_binary(const Mat<double> &x, const Mat<std::string> &y) override;
    Mat<std::string>   predict_binary(const Mat<double> &x, const Mat<double> &theta) const override;

  public:
    std::shared_ptr<ClassificationModelBase<double>> clone() const override;
};
} // namespace TL
#endif // MULTILAYER_PERCEPTRON_CLASSIFICATION_B2M_HPP