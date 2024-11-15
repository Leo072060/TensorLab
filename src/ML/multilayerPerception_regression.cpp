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
        set<size_t>                 samples;
        random_device               rd;
        mt19937                     gen(rd());
        uniform_real_distribution<> dis(0, x.size(Axis::row) - 1);
        while (samples.size() < batch_size)
            samples.insert(dis(gen));

        size_t theLast = neurons.size() - 1;
        double maxLoss = 0;
        for (const auto sample : samples)
        {
            for (size_t i = 0; i < neurons[0].size(); ++i)
            {
                neurons[0][i]->sentSignal(x.iloc(sample, i));
            }
            for (size_t i = 1; i < neurons.size() - 1; ++i)
                for (size_t j = 0; j < neurons[i].size(); ++j)
                {
                    neurons[i][j]->activate();
                    neurons[i][j]->sentSignal();
                }
            for (size_t j = 0; j < neurons[theLast].size(); ++j)
            {
                neurons[theLast][j]->activate();
            }

            // calculate delta
            vector<double> y_target, y_pred;
            for (size_t i = 0; i < neurons[theLast].size(); ++i)
            {
                double y_i_pred   = neurons[theLast][i]->getOutputSignal_outputLayer();
                double y_i_target = y.iloc(sample, i);
                y_pred.emplace_back(y_i_pred);
                y_target.emplace_back(y_i_target);
                neurons[theLast][i]->setDelta_outputLayer(lossFunction_partialDerivative(y_i_target, y_i_pred),
                                                          batch_size);
            }

            double loss = calLoss(y_target, y_pred);
            if (loss > maxLoss) maxLoss = loss;

            for (long long int i = neurons.size() - 2; i >= 0; --i)
                for (auto &e : neurons[i])
                    e->calDelta(batch_size);
            for (long long int i = neurons.size() - 1; i >= 0; --i)
                for (auto &e : neurons[i])
                    e->clearSignals();
        }

        cout << "epoch: " << I << " - max loss: " << maxLoss << endl;
        // cin.get();

        for (long long int i = neurons.size() - 1; i >= 0; --i)
            for (auto &e : neurons[i])
            {
                e->adjust(learning_rate);
                e->resetDelta();
            }
    }

    return neuralNetwork2theta(neurons);
}
Mat<double> MultilayerPerception_regression::predict_(const Mat<double> &x, const Mat<double> &theta) const
{
    using namespace std;

    Mat<double> y(x.size(Axis::row), 1);

    if (theta.iloc(0, 0) != x.size(Axis::col))
    {
        cerr << "The dimension of x does not match the model's requirement." << endl;
        throw runtime_error("Dimension mismatch between x and the model.");
    }

    vector<vector<shared_ptr<neuron>>> neurons_pred = theta2neuralNetwork(theta);

    size_t theLast = neurons_pred.size() - 1;
    for (size_t r = 0; r < x.size(Axis::row); ++r)
    {
        for (auto &e : neurons_pred)
            for (auto &e2 : e)
                e2->clearSignals();
        for (size_t i = 0; i < neurons_pred[0].size(); ++i)
        {
            neurons_pred[0][i]->sentSignal(x.iloc(r, i));
            neurons_pred[0][i]->sentSignal(x.iloc(r, i));
        }
        for (size_t i = 1; i < neurons_pred.size(); ++i)
        {
            for (size_t j = 0; j < neurons_pred[i].size(); ++j)
            {
                neurons_pred[i][j]->activate();
                neurons_pred[i][j]->sentSignal();
            }
        }
        for (size_t i = 0; i < neurons_pred[theLast].size(); ++i)
        {
            neurons_pred[theLast][i]->activate();
            y.iloc(r, i) = neurons_pred[theLast][i]->getOutputSignal_outputLayer();
        }
    }

    return y;
}
double MultilayerPerception_regression::calLoss(const std::vector<double> &y_target,
                                                const std::vector<double> &y_pred) const
{
    using namespace std;

    if (y_target.size() != y_pred.size())
    {
        cerr << "Error: The size of y_target does not match the size of y_pred!" << endl;
        throw std::invalid_argument("Size mismatch between y_target and y_pred.");
    }

    double loss = 0.0;

    switch (lossFunction)
    {
    case LossFunction::MSE: {
        double sum = 0.0;
        size_t n   = y_target.size();
        for (size_t i = 0; i < n; ++i)
        {
            sum += std::pow(y_target[i] - y_pred[i], 2);
        }
        loss = sum / n;
        break;
    }
    default:
        cerr << "Error: Unsupported loss function!" << endl;
        throw std::invalid_argument("Unsupported loss function.");
    }

    return loss;
}
double MultilayerPerception_regression::lossFunction_partialDerivative(const double y_target, const double y_pred) const
{
    using namespace std;

    switch (lossFunction)
    {
    case LossFunction::MSE:
        return -(y_target - y_pred);
    default:
        cerr << "Error: Unsupported loss function!" << endl;
        throw std::invalid_argument("Unsupported loss function.");
    }
}
std::vector<std::vector<std::shared_ptr<MultilayerPerception_regression::neuron>>> MultilayerPerception_regression::
    buildNeuralNetwork(const Mat<double> &x, const Mat<double> &y) const
{
    using namespace std;

    // check architecture
    vector<size_t> architecture = architecture_hiddenLayer;
    if (architecture.empty())
    {
        // default architecture
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
        layer_input.emplace_back(make_shared<neuron>(activation_hidden, 0.5));
    }
    neurons.push_back(layer_input);
    for (size_t i = 1; i < architecture.size() - 1; ++i)
    {
        vector<shared_ptr<neuron>> layer_hidden;
        for (size_t j = 0; j < architecture[i]; ++j)
            layer_hidden.emplace_back(make_shared<neuron>(activation_hidden, 0.0));
        neurons.push_back(layer_hidden);
    }
    vector<shared_ptr<neuron>> layer_output;
    for (size_t i = 0; i < architecture[architecture.size() - 1]; ++i)
    {
        layer_output.emplace_back(make_shared<neuron>(activation_output, 0.0));
    }
    neurons.push_back(layer_output);

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
Mat<double> MultilayerPerception_regression::neuralNetwork2theta(
    const std::vector<std::vector<std::shared_ptr<neuron>>> &neurons) const
{
    using namespace std;

    size_t maxSize = 0;
    for (size_t i = 0; i < neurons.size() - 1; ++i)
    {
        double size = (neurons[i].size() * (neurons[i + 1].size() + 1)) + 1;
        if (size > maxSize) maxSize = size;
    }
    Mat<double> theta(maxSize, neurons.size());

    for (size_t i = 0; i < neurons.size(); ++i)
    {
        theta.iloc(0, i) = neurons[i].size();
        for (size_t j = 0; j < neurons[i].size(); ++j)
        {
            theta.iloc(j + 1, i) = neurons[i][j]->getThreshold();
            for (size_t k = 0; k < neurons[i][j]->synapsesSize(); ++k)
            {
                theta.iloc(5 * j + 6 + k, i) = neurons[i][j]->getSynapse(k).weight;
            }
        }
    }
    for (size_t i = 0; i < neurons.size(); ++i)
    {
        string activationType = "";
        switch (neurons[i][0]->getActivationType())
        {
        case Activation::equation:
            activationType = "equation";
            break;
        case Activation::sigmoid:
            activationType = "sigmoid";
        case Activation::tanh:
            activationType = "tanh";
            break;
        default:
            cerr << "Error: Unsupported activation function!" << endl;
            throw std::invalid_argument("Unsupported activation function.");
        }
        theta.iloc_name(i, Axis::col) = activationType;
    }

    string lossFunctionType = "";
    switch (lossFunction)
    {
    case LossFunction::MSE:
        lossFunctionType = "MSE";
        break;
    default:
        cerr << "Error: Unsupported loss function!" << endl;
        throw std::invalid_argument("Unsupported loss function.");
    }
    theta.iloc_name(0, Axis::row) = lossFunctionType;

    return theta;
}
std::vector<std::vector<std::shared_ptr<MultilayerPerception_regression::neuron>>> MultilayerPerception_regression::
    theta2neuralNetwork(const Mat<double> &theta) const
{
    using namespace std;

    vector<vector<shared_ptr<neuron>>> neurons;

    for (size_t c = 0; c < theta.size(Axis::col); ++c)
    {
        Activation activationType;
        if (theta.iloc_name(c, Axis::col) == "equation")
            activationType = Activation::equation;
        else if (theta.iloc_name(c, Axis::col) == "sigmoid")
            activationType = Activation::sigmoid;
        else if (theta.iloc_name(c, Axis::col) == "tanh")
            activationType = Activation::tanh;

        vector<shared_ptr<neuron>> layer;
        for (size_t i = 1; i <= theta.iloc(0, c); ++i)
            layer.emplace_back(make_shared<neuron>(activationType, theta.iloc(i, c)));
        neurons.emplace_back(layer);
    }
    for (size_t i = 0; i < neurons.size() - 1; ++i)
        for (size_t j = 0; j < neurons[i].size(); ++j)
            for (size_t k = 0; k < neurons[i + 1].size(); ++k)
                neurons[i][j]->connect(theta.iloc(5 * j + 6 + k, i), neurons[i + 1][k]);

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
    : inputSignal(0.0)
    , outputSignal(0.0)
    , threshold(initialThreshold)
    , synapses()
    , delta(0.0)
    , threshold_delta(0.0)
    , activationType(type)
{
}
void MultilayerPerception_regression::neuron::connect(const double w, std::shared_ptr<neuron> other)
{
    Synapse syn;
    syn.link         = other;
    syn.weight       = 0;
    syn.weight_delta = 0;
    synapses.push_back(syn);
}
double MultilayerPerception_regression::neuron::getOutputSignal_outputLayer() const
{
    return outputSignal;
}
void MultilayerPerception_regression::neuron::setDelta_outputLayer(const double d, const double batchSize)
{
    delta = d * activation_derivative(inputSignal - threshold);
    threshold_delta += d / batchSize;
}
void MultilayerPerception_regression::neuron::activate()
{
    double x     = inputSignal - threshold;
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
    case Activation::sigmoid:
        return activation(x) * (1 - activation(x));
    case Activation::equation:
        return 1;
    case Activation::tanh:
        return 1 - activation(x);
    default:
        cerr << "Error: Unsupported activation function!" << endl;
        throw std::invalid_argument("Unsupported activation function.");
    }
}
void MultilayerPerception_regression::neuron::sentSignal() const
{
    for (auto &e : synapses)
        e.link->inputSignal += e.weight * outputSignal;
}
void MultilayerPerception_regression::neuron::sentSignal(const double signal)
{
    outputSignal = signal;
    sentSignal();
}
void MultilayerPerception_regression::neuron::calDelta(const size_t batchSize)
{
    delta = 0;
    for (size_t i = 0; i < synapses.size(); ++i)
    {
        delta += synapses[i].weight * synapses[i].link->delta;
        synapses[i].weight_delta -= synapses[i].link->delta * outputSignal / batchSize;
    }
    delta *= activation_derivative(inputSignal - threshold);
    threshold_delta += delta / batchSize;
}
void MultilayerPerception_regression::neuron::adjust(const double learningRate)
{
    for (size_t i = 0; i < synapses.size(); ++i)
    {
        synapses[i].weight += learningRate * synapses[i].weight_delta;
    }
    threshold += learningRate * threshold_delta;
}
void MultilayerPerception_regression::neuron::resetDelta()
{
    delta = 0;
    for (size_t i = 0; i < synapses.size(); ++i)
    {
        synapses[i].weight_delta = 0;
    }
    threshold_delta = 0;
}
void MultilayerPerception_regression::neuron::clearSignals()
{
    inputSignal  = 0;
    outputSignal = 0;
}
double MultilayerPerception_regression::neuron::getThreshold() const
{
    return threshold;
}
size_t MultilayerPerception_regression::neuron::synapsesSize() const
{
    return synapses.size();
}
MultilayerPerception_regression::neuron::Synapse MultilayerPerception_regression::neuron::getSynapse(
    const size_t i) const
{
    using namespace std;

    if (i >= synapses.size())
    {
        cerr << "Error: Index " << i << " is out of range for synapses." << endl;
        throw out_of_range("Index is out of range in getSynapse.");
    }
    return synapses[i];
}
MultilayerPerception_regression::Activation MultilayerPerception_regression::neuron::getActivationType() const
{
    return activationType;
}