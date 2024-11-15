#include <limits>
#include <random>

#include "ML/multilayerPerceptron_classification.hpp"
#include "preprocessor/encode.hpp"

using namespace TL;

Mat<double> MultilayerPerception_classification::train_multi(const Mat<double> &x, const Mat<std::string> &y)
{
    using namespace std;

    if (x.size(Axis::row) < batch_size)
    {
        cerr << "Error: Batch size (" << batch_size << ") is larger than the available rows (" << x.size(Axis::row)
             << ")." << endl;
        throw out_of_range("Batch size is larger than the available rows.");
    }

    OneHotEncoder encoder;
    Mat<double>   y_oneHot = encoder.fit_transform(y);
    record(managed_labels, encoder.get_labels());

    vector<vector<shared_ptr<neuron>>> neurons = buildNeuralNetwork(x, y_oneHot);

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
        double maxLoss = -std::numeric_limits<double>::infinity();
        for (const auto sample : samples)
        {
            for (const auto &e : neurons)
                for (auto &e2 : e)
                    e2->clearSignals();

            for (size_t i = 0; i < neurons[0].size(); ++i)
            {
                neurons[0][i]->sentSignal(x.iloc(sample, i));
            }
            for (size_t i = 1; i < neurons.size() - 1; ++i)
            {
                for (size_t j = 0; j < neurons[i].size(); ++j)
                {
                    neurons[i][j]->activate();
                    neurons[i][j]->sentSignal();
                }
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
                double y_i_target = y_oneHot.iloc(sample, i);
                y_pred.emplace_back(y_i_pred);
                y_target.emplace_back(y_i_target);
                neurons[theLast][i]->setDelta_outputLayer(lossFunction_partialDerivative(y_i_target, y_i_pred),
                                                          batch_size);
            }

            // DEBUG
            // for (auto e : y_target)
            //     cout << e << " ";
            // cout << endl;
            // for (auto e : y_pred)
            //     cout << e << " ";
            // cout << endl;
            // cin.get();

            double loss = calLoss(y_target, y_pred);
            if (loss > maxLoss) maxLoss = loss;

            for (long long int i = neurons.size() - 2; i >= 0; --i)
                for (auto &e : neurons[i])
                    e->calDelta(batch_size);
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

    neuralNetwork = neurons;
    return Mat<double>();
}
Mat<std::string> MultilayerPerception_classification::predict_multi(const Mat<double> &x,
                                                                    const Mat<double> &theta) const
{
    using namespace std;

    Mat<double> y_possible(x.size(Axis::row), this->managed_labels.read().size(Axis::col));

    // if (theta.iloc(0, 0) != x.size(Axis::col))
    // {
    //     cerr << "The dimension of x does not match the model's requirement." << endl;
    //     throw runtime_error("Dimension mismatch between x and the model.");
    // }

    vector<vector<shared_ptr<neuron>>> neurons_pred = neuralNetwork;

    size_t theLast = neurons_pred.size() - 1;
    for (size_t r = 0; r < x.size(Axis::row); ++r)
    {
        for (const auto &e : neurons_pred)
            for (auto &e2 : e)
                e2->clearSignals();
        for (size_t i = 0; i < neurons_pred[0].size(); ++i)
        {
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
            y_possible.iloc(r, i) = neurons_pred[theLast][i]->getOutputSignal_outputLayer();
        }
    }

    Mat<int> y_oneHot(x.size(Axis::row), this->managed_labels.read().size(Axis::col));
    for (size_t r = 0; r < y_possible.size(Axis::row); ++r)
    {
        double mostPossible = 0;
        size_t index_oneHot = 0;
        for (size_t c = 0; c < y_possible.size(Axis::col); ++c)
        {
            if (y_possible.iloc(r, c) > mostPossible)
            {
                mostPossible = y_possible.iloc(r, c);
                index_oneHot = c;
            }
        }
        y_oneHot.iloc(r, index_oneHot) = 1;
    }

    display(y_possible);
    cin.get();

    OneHotEncoder encoder;
    encoder.set_labels(managed_labels.read());
    Mat<string> y = encoder.decode(y_oneHot);

    return y;
}
double MultilayerPerception_classification::calLoss(const std::vector<double> &y_target,
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
    case LossFunction::CrossEntropy: {
        size_t n = y_target.size();
        for (size_t i = 0; i < n; ++i)
        {
            // Avoid log(0) by checking for small values in y_pred.
            if (y_pred[i] > 0)
            {
                loss -= y_target[i] * std::log(y_pred[i]);
            }
            else
            {
                cerr << "Error: y_pred is 0, this will cause division by zero." << endl;
                throw invalid_argument("Division by zero.");
            }
        }
        break;
    }
    default:
        cerr << "Error: Unsupported loss function!" << endl;
        throw std::invalid_argument("Unsupported loss function.");
    }

    return loss;
}
double MultilayerPerception_classification::lossFunction_partialDerivative(const double y_target,
                                                                           const double y_pred) const
{
    using namespace std;

    switch (lossFunction)
    {
    case LossFunction::MSE:
        return -(y_target - y_pred);
    case LossFunction::CrossEntropy: {
        if (y_pred == 0)
        {
            cerr << "Error: y_pred is 0, this will cause division by zero." << endl;
            throw invalid_argument("Division by zero.");
        }
        return -(y_target / y_pred);
    }
    default:
        cerr << "Error: Unsupported loss function!" << endl;
        throw invalid_argument("Unsupported loss function.");
    }
}
std::vector<std::vector<std::shared_ptr<MultilayerPerception_classification::neuron>>>
MultilayerPerception_classification::buildNeuralNetwork(const Mat<double> &x, const Mat<double> &y)
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

// for polymorphism
std::shared_ptr<ClassificationModelBase<double>> MultilayerPerception_classification ::clone() const
{
    using namespace std;

    return make_shared<MultilayerPerception_classification>(*this);
}

// neuron
MultilayerPerception_classification::neuron::neuron(const Activation type, const double initialThreshold)
    : inputSignal(0.0)
    , outputSignal(0.0)
    , threshold(initialThreshold)
    , synapses()
    , delta(0.0)
    , threshold_delta(0.0)
    , activationType(type)
{
}
void MultilayerPerception_classification::neuron::connect(const double w, std::shared_ptr<neuron> other)
{
    Synapse syn;
    syn.link         = other;
    syn.weight       = w;
    syn.weight_delta = 0;
    synapses.push_back(syn);
}
double MultilayerPerception_classification::neuron::getOutputSignal_outputLayer() const
{
    return outputSignal;
}
void MultilayerPerception_classification::neuron::setDelta_outputLayer(const double d, const double batchSize)
{
    delta = d * activation_derivative(inputSignal - threshold);
    threshold_delta += d / batchSize;
}
void MultilayerPerception_classification::neuron::activate()
{
    double x     = inputSignal - threshold;
    outputSignal = activation(x);
}
double MultilayerPerception_classification::neuron::activation(const double x) const
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
double MultilayerPerception_classification::neuron::activation_derivative(const double x) const
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
void MultilayerPerception_classification::neuron::sentSignal() const
{
    for (auto &e : synapses)
        e.link->inputSignal += e.weight * outputSignal;
}
void MultilayerPerception_classification::neuron::sentSignal(const double signal)
{
    outputSignal = signal;
    sentSignal();
}
void MultilayerPerception_classification::neuron::calDelta(const size_t batchSize)
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
void MultilayerPerception_classification::neuron::adjust(const double learningRate)
{
    for (size_t i = 0; i < synapses.size(); ++i)
    {
        synapses[i].weight += learningRate * synapses[i].weight_delta;
    }
    threshold += learningRate * threshold_delta;
}
void MultilayerPerception_classification::neuron::resetDelta()
{
    delta = 0;
    for (size_t i = 0; i < synapses.size(); ++i)
    {
        synapses[i].weight_delta = 0;
    }
    threshold_delta = 0;
}
void MultilayerPerception_classification::neuron::clearSignals()
{
    inputSignal  = 0;
    outputSignal = 0;
}
