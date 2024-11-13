#include <random>

#include "ML/multilayerPerceptron_regression.hpp"

using namespace TL;

Mat<double> MultilayerPerception_regression::train_(const Mat<double> &x, const Mat<double> &y)
{
    using namespace std;

    if (x.size(Axis::row) < batch_size)
    {
        cerr << "Error: Batch size (" << batch_size << ") is larger than the available rows (" << x.size(Axis::row)
             << ")." << endl;
        throw out_of_range("Batch size is larger than the available rows.");
    }

    vector<vector<shared_ptr<neuron>>> neurons = buildNeuralNetwork(x, y);

    // start training
    for (size_t I = 0; I < iterations; ++I)
    {
        // randomly select samples to train
        set<size_t>                samples;
        random_device              rd;
        mt19937                    gen(rd());
        uniform_int_distribution<> dis(0, x.size(Axis::row) - 1);
        while (samples.size() < batch_size)
            samples.insert(dis(gen));

        for (const auto sample : samples)
        {
            for (size_t i = 0; i < neurons[0].size(); ++i)
            {
                neurons[0][i]->sentSignal(x.iloc(sample, i));
            }
            for (size_t i = 1; i < neurons.size(); ++i)
                for (size_t j = 0; j < neurons[i].size(); ++j)
                {
                    neurons[i][j]->activate();
                    neurons[i][j]->sentSignal();
                }

            // adjust
            size_t theLast = neurons.size() - 1;
            for (size_t i = 0; i < neurons[theLast].size(); ++i)
            {
                double y_i_pred = neurons[theLast][i]->getOutputSignal();
                double delta_i  = (y_i_pred - y.iloc(sample, i)) * y_i_pred * (1 - y_i_pred);
                neurons[theLast][i]->initOutputLayerDeltaAndAdjustThreshold(delta_i, learning_rate);
            }
            for (size_t i = neurons.size() - 2; i >= 0; --i)
            {
                for (auto &e : neurons[i])
                {
                    e->adjust(learning_rate);
                    e->clearSignals();
                }
            }
        }
    }

    neuralNetwork = neurons;

    return Mat<double>();
}
Mat<double> MultilayerPerception_regression::predict_(const Mat<double> &x, const Mat<double> &theta) const
{
    using namespace std;

    Mat<double> y(x.size(Axis::row), theta.iloc(0, theta.size(Axis::col) - 1));

    // if (theta.iloc(0, 0) != x.size(Axis::col))
    // {
    //     cerr << "The dimension of x does not match the model's requirement." << endl;
    //     throw runtime_error("Dimension mismatch between x and the model.");
    // }

    vector<vector<shared_ptr<neuron>>> neurons_pred = neuralNetwork;

    for (size_t r = 0; r < x.size(Axis::row); ++r)
    {
        for (size_t i = 0; i < neurons_pred[0].size(); ++i)
        {
            neurons_pred[0][i]->sentSignal(x.iloc(r, i));
        }
        for (size_t i = 1; i < neurons_pred.size() - 1; ++i)
            for (size_t j = 0; j < neurons_pred[i].size(); ++j)
            {
                neurons_pred[i][j]->activate();
                neurons_pred[i][j]->sentSignal();
            }
        size_t theLast = neurons_pred.size() - 1;
        for (size_t i = 0; i < neurons_pred[theLast].size(); ++i)
        {
            neurons_pred[i][theLast]->activate();
            y.iloc(r, i) = neurons_pred[i][theLast]->getOutputSignal();
        }
    }

    return y;
}
std::vector<std::vector<std::shared_ptr<MultilayerPerception_regression::neuron>>> MultilayerPerception_regression::
    buildNeuralNetwork(const Mat<double> &x, const Mat<double> &y)
{
    using namespace std;

    // check architecture
    vector<size_t> architecture = architecture_hiddenLayer;
    if (architecture.empty())
    {
        // default architecture
        architecture.emplace_back(x.size(Axis::col));
        architecture.emplace_back(x.size(Axis::col));
        architecture.emplace_back(x.size(Axis::col));
    }
    architecture.insert(architecture.begin(), x.size(Axis::col));
    architecture.emplace_back(y.size(Axis::col));
    for (const auto &layer_size : architecture)
    {
        if (layer_size == 0)
        {
            cerr << "Error: Architecture vector contains a layer with zero neurons." << endl;
            throw runtime_error("Each layer must have at least one neuron.");
        }
    }

    vector<vector<shared_ptr<neuron>>> neurons;

    // create neurons
    vector<shared_ptr<neuron>> layer_input;
    for (size_t i = 0; i < architecture[0]; ++i)
    {
        layer_input.emplace_back(make_shared<neuron>(activation_hidden, 0.0));
        neurons.push_back(layer_input);
    }
    for (const auto e : architecture)
    {
        vector<shared_ptr<neuron>> layer_hidden;
        for (size_t i = 0; i < e; ++i)
            layer_hidden.emplace_back(make_shared<neuron>(activation_hidden, 0.0));
        neurons.push_back(layer_hidden);
    }
    vector<shared_ptr<neuron>> layer_output;
    for (size_t i = 0; i < architecture[architecture.size() - 1]; ++i)
    {
        layer_input.emplace_back(make_shared<neuron>(activation_output, 0.0));
        neurons.push_back(layer_output);
    }

    // connect neurons
    random_device rd;
    mt19937       gen(rd());
    for (size_t i = 0; i < neurons.size() - 1; ++i)
    {
        // Use Xavier to init weight for each synapse.
        double                      range = sqrt(2.0 / (neurons[i].size() + neurons[i + 1].size()));
        uniform_real_distribution<> dis(-range, range);
        for (auto &e : neurons[i])
            for (const auto &e_nextLayer : neurons[i + 1])
                e->connect(dis(gen), e_nextLayer);
    }

    return neurons;
}

// for polymorphism
std::shared_ptr<RegressionModelBase<double>> MultilayerPerception_regression ::clone() const
{
    using namespace std;

    return make_shared<MultilayerPerception_regression>(*this);
}

// neuron
MultilayerPerception_regression::neuron::neuron(const Activation type, const double initialThreshold)
    : activationType(type)
    , threshold(initialThreshold)
{
}
void MultilayerPerception_regression::neuron::connect(const double weight, std::shared_ptr<neuron> other)
{
    synapse.push_back({weight, other});
}
void MultilayerPerception_regression::neuron::receiveSignal(const double x)
{
    inputSignal += x;
}
double MultilayerPerception_regression::neuron::getOutputSignal() const
{
    return outputSignal;
}
void MultilayerPerception_regression::neuron::initOutputLayerDeltaAndAdjustThreshold(const double d,
                                                                                     const double learning_rate)
{
    delta = d;
    threshold -= learning_rate * delta;
}
void MultilayerPerception_regression::neuron::activate()
{
    double x     = threshold - inputSignal;
    outputSignal = activation(x);
}
double MultilayerPerception_regression::neuron::activation(const double x) const
{
    using namespace std;

    switch (activationType)
    {
    case Activation::equation:
        return x;
    case Activation::tanh:
        return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
    case Activation::sigmoid:
        return 1 / (1 + exp(-x));
    default:
        cerr << "Error: Unsupported activation function!" << endl;
        throw std::invalid_argument("Unsupported activation function.");
    }
}
double MultilayerPerception_regression::neuron::activation_derivative(const double x) const
{
    using namespace std;

    switch (activationType)
    {
    case Activation::equation:
        return 1;
    case Activation::tanh:
        return 1 - activation(x);
    case Activation::sigmoid:
        return activation(x) * (1 - activation(x));
    default:
        cerr << "Error: Unsupported activation function!" << endl;
        throw std::invalid_argument("Unsupported activation function.");
    }
}
void MultilayerPerception_regression::neuron::sentSignal() const
{
    for (auto &e : synapse)
        e.second->inputSignal += e.first * outputSignal;
}
void MultilayerPerception_regression::neuron::sentSignal(const double signal)
{
    outputSignal = signal;
    sentSignal();
}
void MultilayerPerception_regression::neuron::adjust(const double learning_rate)
{
    delta = 0;
    for (size_t i = 0; i < synapse.size(); ++i)
    {
        double delta_i = 0;
        delta_i        = synapse[i].first * synapse[i].second->delta * activation_derivative(inputSignal);
        delta += delta_i * synapse[i].first;
        synapse[i].first += learning_rate * synapse[i].second->delta * outputSignal;
    }
    threshold -= learning_rate * delta;
}
void MultilayerPerception_regression::neuron::clearSignals()
{
    inputSignal  = 0;
    outputSignal = 0;
}