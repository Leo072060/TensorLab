#ifndef MULTILAYER_PERCEPTRON_REGRESSION
#define MULTILAYER_PERCEPTRON_REGRESSION

#include "ML/_internal/regressionModelBase.hpp"

namespace TL
{
using namespace _internal;

class MultilayerPerception_regression : public RegressionModelBase<double>
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
    double              learning_rate = 0.0003;
    size_t              batch_size    = 100;
    size_t              iterations    = 1700;
    std::vector<size_t> architecture_hiddenLayer;

  private:
    class neuron
    {
      public:
        double inputSignal;
        double outputSignal;
        double threshold;
        void   activate();
        void   sentSignal();
        void   calAdjustVal(const double learning);
        void   adjust();
        void   clearSignals();

        std::vector<std::pair<double, std::shared_ptr<neuron>>> synapse;

        double              g; // g is used for last neurons to adjust
        std::vector<double> weights_adjusted;
        double              threshold_adjusted;
    };

    std::vector<std::vector<std::shared_ptr<neuron>>> buildNetwork(const Mat<double> &x, const Mat<double> &y);
    static Mat<double> neuralNetwork2theta(const std::vector<std::vector<std::shared_ptr<neuron>>> neurons);
    static std::vector<std::vector<std::shared_ptr<neuron>>> theta2neuralNetwork(const Mat<double> &theta);
};
} // namespace TL
#endif // MULTILAYER_PERCEPTRON_REGRESSION