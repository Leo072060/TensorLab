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
    enum LossFunction
    {
        MSE
    };

    double              learning_rate     = 0.03;
    Activation          activation_hidden = Activation::sigmoid;
    Activation          activation_output = Activation::sigmoid;
    LossFunction        lossFunction      = LossFunction::MSE;
    size_t              batch_size        = 200;
    size_t              iterations        = 60000;
    double              tolerance         = 0.1;
    std::vector<size_t> architecture_hiddenLayer;

  private:
    double calLoss(const std::vector<double> &y_target, const std::vector<double> &y_pred) const;
    double lossFunction_partialDerivative(const double y_target, const double y_pred) const;

    class neuron
    {
      public:
        neuron(const Activation type, const double initialThreshold);

        void   connect(const double w, std::shared_ptr<neuron> other);
        double getOutputSignal_outputLayer() const;
        void   setDelta_outputLayer(const double d,const double batchSize);
        void   activate();
        double activation(const double x) const;
        double activation_derivative(const double x) const;
        void   sentSignal() const;
        void   sentSignal(const double signal);
        void   calDelta(const size_t batchSize);
        void   adjust(const double learningRate);
        void   resetDelta();
        void   clearSignals();

      private:
        double inputSignal;
        double outputSignal;

        double threshold;

        struct Synapse
        {
            std::shared_ptr<neuron> link;
            double                  weight;
            double                  weight_delta;
        };
        std::vector<Synapse> synapses;

        double delta;
        double threshold_delta;

        // neuron parameters
        Activation activationType;
    };

    std::vector<std::vector<std::shared_ptr<neuron>>> buildNeuralNetwork(const Mat<double> &x, const Mat<double> &y);
    std::vector<std::vector<std::shared_ptr<neuron>>> neuralNetwork;
};
} // namespace TL
#endif // MULTILAYER_PERCEPTRON_REGRESSION