#include <random>

#include "ML/multilayerPerceptron_classification.hpp"
#include "preprocessor/encode.hpp"

using namespace TL;

Mat<double> MultilayerPerception_classification::train_multi(const Mat<double> &x, const Mat<std::string> &y)
{
    using namespace std;

    Mat<double>                        y_encode = onehot_encode(y);
    vector<vector<shared_ptr<neuron>>> neurons  = buildNetwork(x, y_encode);

    // start training
    if (x.size(Axis::row) < batch_size)
    {
        cerr << "Error: Batch size (" << batch_size << ") is larger than the available rows (" << x.size(Axis::row)
             << ")." << endl;
        throw out_of_range("Batch size is larger than the available rows.");
    }
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
                neurons[0][i]->outputSignal = x.iloc(sample, i);
                neurons[0][i]->sentSignal();
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
                double y_i_pred        = neurons[theLast][i]->outputSignal;
                neurons[theLast][i]->g = (y_i_pred - y_encode.iloc(sample, i)) * y_i_pred * (1 - y_i_pred);
            }
            for (size_t i = 0; i < neurons.size() - 1; ++i)
            {
                for (auto &e : neurons[i])
                    e->calAdjustVal(learning_rate);
            }
            for (size_t i = 0; i < neurons.size() - 1; ++i)
            {
                for (auto &e : neurons[i])
                {
                    e->adjust();
                    e->clearSignals();
                }
            }
        }
    }

    return neuralNetwork2theta(neurons);
}
Mat<std::string> MultilayerPerception_classification::predict_multi(const Mat<double> &x,
                                                                    const Mat<double> &theta) const
{
    using namespace std;

    Mat<double> y_encode(x.size(Axis::row), theta.iloc(0, theta.size(Axis::col) - 1));
    Mat<string> y(x.size(Axis::row), 1);

    if (theta.iloc(0, 0) != x.size(Axis::col))
    {
        cerr << "The dimension of x does not match the model's requirement." << endl;
        throw runtime_error("Dimension mismatch between x and the model.");
    }

    vector<vector<shared_ptr<neuron>>> neurons_pred = theta2neuralNetwork(theta);
    vector<string>                     types;
    for (size_t i = 0; i < theta.iloc(0, theta.size(Axis::col) - 1); ++i)
        types.emplace_back(theta.iloc_name(i, Axis::row));

    for (size_t r = 0; r < x.size(Axis::row); ++r)
    {
        for (size_t i = 0; i < neurons_pred[0].size(); ++i)
        {
            neurons_pred[0][i]->outputSignal = x.iloc(r, i);
            neurons_pred[0][i]->sentSignal();
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
            y_encode.iloc(r, i) = neurons_pred[i][theLast]->outputSignal;
        }
    }

    for (size_t r = 0; r < y_encode.size(Axis::row); ++r)
    {
        double highestPossible = 0;
        size_t index           = 0;
        for (size_t c = 0; c < y_encode.size(Axis::col); ++c)
        {
            if (y_encode.iloc(r, c) > highestPossible)
            {
                highestPossible = y_encode.iloc(r, c);
                index           = c;
            }
        }
        y.iloc(r, 0) = types[index];
    }

    return y;
}
std::vector<std::vector<std::shared_ptr<MultilayerPerception_classification::neuron>>>
MultilayerPerception_classification::buildNetwork(const Mat<double> &x, const Mat<double> &y)
{
    using namespace std;

    // check architecture
    vector<size_t> architecture = architecture_hiddenLayer;
    if (architecture.empty())
    {
        // default architecture
        architecture.emplace_back(x.size(Axis::col));
        architecture.emplace_back(x.size(Axis::col));
    }
    architecture.insert(architecture.begin(), x.size(Axis::col));
    architecture.push_back(y.size(Axis::col));
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
    for (const auto e : architecture)
    {
        vector<shared_ptr<neuron>> layer;
        for (size_t i = 0; i < e; ++i)
            layer.emplace_back(make_shared<neuron>());
        neurons.push_back(layer);
    }

    // init neurons and connect neurons
    random_device rd;
    mt19937       gen(rd());
    for (size_t i = 0; i < neurons.size() - 1; ++i)
    {
        // Use Xavier to init weight for each synapse.
        double                     range = pow(1 / neurons[i].size(), 0.5);
        uniform_int_distribution<> dis(-range, range);
        for (auto &e : neurons[i])
        {
            e->threshold = 0.0;
            for (const auto &e_nextLayer : neurons[i + 1])
                e->synapse.push_back({dis(gen), e_nextLayer});
        }
    }
    for (auto &e : neurons[neurons.size() - 1])
        e->threshold = 0.0;

    return neurons;
}
Mat<double> MultilayerPerception_classification::neuralNetwork2theta(
    const std::vector<std::vector<std::shared_ptr<neuron>>> neurons, const Mat<std::string> &y_unique)
{
    size_t maxSize = 0;
    for (size_t i = 0; i < neurons.size() - 1; ++i)
    {
        double size = (neurons[i].size() * (neurons[i + 1].size() + 1)) + 1;
        if (size > maxSize) maxSize = size;
    }
    Mat<double> theta(maxSize, neurons.size());

    for (size_t i = 0; i < y_unique.size(Axis::col); ++i)
        theta.iloc_name(i, Axis::row) = y_unique.iloc(0, i);
    for (size_t i = 0; i < neurons.size(); ++i)
    {
        theta.iloc(0, i) = neurons[i].size();
        for (size_t j = 0; j < neurons[i].size(); ++j)
        {
            theta.iloc(j + 1, i) = neurons[i][j]->threshold;
            for (size_t k = 0; k < neurons[i][j]->synapse.size(); ++k)
            {
                theta.iloc(5 * j + 6 + k, i) = neurons[i][j]->synapse[k].first;
            }
        }
    }

    return theta;
}
std::vector<std::vector<std::shared_ptr<MultilayerPerception_classification::neuron>>>
MultilayerPerception_classification::theta2neuralNetwork(const Mat<double> &theta)
{
    using namespace std;

    vector<vector<shared_ptr<neuron>>> neurons;

    for (size_t c = 0; c < theta.size(Axis::col); ++c)
    {
        vector<shared_ptr<neuron>> layer;
        for (size_t i = 0; i < theta.iloc(0, c); ++i)
            layer.emplace_back(make_shared<neuron>());
    }
    for (size_t i = 0; i < neurons.size() - 1; ++i)
        for (size_t j = 0; j < neurons[i].size(); ++j)
            for (size_t k = 0; k < neurons[i + 1].size(); ++k)
                neurons[i][j]->synapse.push_back({theta.iloc(5 * j + 6 + k, i), neurons[i + 1][k]});

    return neurons;
}

// for polymorphism
std::shared_ptr<ClassificationModelBase<double>> MultilayerPerception_classification ::clone() const
{
    using namespace std;

    return make_shared<MultilayerPerception_classification>(*this);
}

// neuron
void MultilayerPerception_classification::neuron::activate()
{
    outputSignal = 1 / (1 + exp(threshold - inputSignal));
}
void MultilayerPerception_classification::neuron::sentSignal()
{
    for (auto &e : synapse)
        e.second->inputSignal += e.first * outputSignal;
}
void MultilayerPerception_classification::neuron::calAdjustVal(const double learning_rate)
{
    weights_adjusted.clear();
    g = 0;
    for (const auto &e : synapse)
    {
        weights_adjusted.push_back(learning_rate * e.second->g * outputSignal);
        g += (e.first * e.second->g);
    }
    g *= (threshold * (1 - threshold));
    threshold_adjusted = -learning_rate * g;
}
void MultilayerPerception_classification::neuron::adjust()
{
    for (size_t i = 0; i < synapse.size(); ++i)
        synapse[i].first = weights_adjusted[i];

    threshold = threshold_adjusted;
}
void MultilayerPerception_classification::neuron::clearSignals()
{
    inputSignal  = 0;
    outputSignal = 0;
}