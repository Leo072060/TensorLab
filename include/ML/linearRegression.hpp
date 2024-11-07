#ifndef LINEAR_REGRESSION_HPP
#define LINEAR_REGRESSION_HPP

#include "ML/_internal/regressionModelBase.hpp"

namespace TL
{
using namespace _internal;

class LinearRegression : public RegressionModelBase<double>
{
    // hook functions
  private:
    Mat<double> train_(const Mat<double> &x, const Mat<double> &y) override;
    Mat<double> predict_(const Mat<double> &x, const Mat<double> &theta) const override;

    // for polymorphism
  public:
    std::shared_ptr<RegressionModelBase<double>> clone() const override;

  public:
    // model parameters
    double learning_rate = 0.0003;
    size_t batch_size    = 100;
    size_t iterations    = 1700;
};
} // namespace TL
#endif // LINEAR_REGRESSION_HPP