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
    enum Activation
    {
        sigmoid,
        tanh,
        equation
    };
    double              learning_rate = 0.0003;
    double              decay_rate;
    double              increase_rate;
    double              decay_rate_grad_explosion;
    double              early_stopping_threshold;
    double              threshold_sustain_count;
    Activation          activation_hidden = Activation::tanh;
    Activation          activation_output = Activation::equation;
    size_t              batch_size = 50 ;
    size_t              iterations = 777;
    std::vector<size_t> architecture_hiddenLayer;

  private:
    class neuron
    {
      public:
        neuron(const Activation type, const double initialThreshold);

        void   connect(const double weight, std::shared_ptr<neuron> other);
        void   receiveSignal(const double x);
        double getOutputSignal() const;
        void   initOutputLayerDeltaAndAdjustThreshold(const double d,const double learning_rate);
        void   activate();
        double activation(const double x) const;
        double activation_derivative(const double x) const;
        void   sentSignal() const;
        void   sentSignal(const double signal);
        void   adjust(const double learning_rate);
        void   clearSignals();

      private:
        double inputSignal;
        double outputSignal;
        double threshold;

        std::vector<std::pair<double, std::shared_ptr<neuron>>> synapse;

        double delta;
        // neuron parameters
        Activation activationType;
    };

    std::vector<std::vector<std::shared_ptr<neuron>>> buildNeuralNetwork(const Mat<double> &x, const Mat<double> &y);
    std::vector<std::vector<std::shared_ptr<neuron>>> neuralNetwork;
};
} // namespace TL
#endif // MULTILAYER_PERCEPTRON_REGRESSION