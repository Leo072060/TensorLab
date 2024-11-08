#ifndef MULTILAYER_PERCEPTRON_CLASSIFICATION
#define MULTILAYER_PERCEPTRON_CLASSIFICATION

#include "ML/_internal/multiClassificationModelBase.hpp"

namespace TL
{
using namespace _internal;

class MultilayerPerception_classification : public MultiClassificationModelBase<double>
{
    // hook functions
  private:
    Mat<double>      train_multi(const Mat<double> &x, const Mat<std::string> &y) override;
    Mat<std::string> predict_multi(const Mat<double> &x, const Mat<double> &theta) const override;

    // for polymorphism
  public:
    std::shared_ptr<ClassificationModelBase<double>> clone() const override;

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
    static Mat<double> neuralNetwork2theta(const std::vector<std::vector<std::shared_ptr<neuron>>> neurons,
                                           const Mat<std::string>                                 &y_unique);
    static std::vector<std::vector<std::shared_ptr<neuron>>> theta2neuralNetwork(const Mat<double> &theta);
};
} // namespace TL
#endif // MULTILAYER_PERCEPTRON_CLASSIFICATION