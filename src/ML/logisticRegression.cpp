#include <random>

#include "ML/logisticRegression.hpp"

using namespace TL;

Mat<double> LogisticRegression::train_binary(const Mat<double> &x, const Mat<std::string> &y)
{
    using namespace std;

    Mat<string> labels = y.unique();
    if (labels.size() != 2)
    {
        cerr << "Error: The target variable must have exactly 2 unique labels for binary classification." << endl;
        throw invalid_argument("Invalid number of labels for binary classification.");
    }

    Mat<double> mumerical_y(y.size(Axis::row), y.size(Axis::col));
    for (size_t i = 0; i < y.size(Axis::row); ++i)
        mumerical_y.iloc(i, 0) = (labels.iloc(0, 0) == y.iloc(i, 0) ? 1 : 0);

    Mat<double> ones(x.size(Axis::row), 1);
    ones          = 1;
    Mat<double> w = x.concat(ones, Axis::col);
    Mat<double> theta(1, x.size(Axis::col) + 1);

    // start training
    for (size_t I = 0; I < iterations; ++I)
    {
        if (x.size(Axis::row) < batch_size)
        {
            cerr << "Error: Batch size (" << batch_size << ") is larger than the available rows (" << x.size(Axis::row)
                 << ")." << endl;
            throw out_of_range("Batch size is larger than the available rows.");
        }

        // generate random numbers
        set<size_t>                randomNums;
        random_device              rd;
        mt19937                    gen(rd());
        uniform_int_distribution<> dis(0, x.size(Axis::row) - 1);
        while (randomNums.size() < batch_size)
            randomNums.insert(dis(gen));

        Mat<double> first_derivative(theta.size(Axis::row), theta.size(Axis::col));
        double      second_derivative = 0;
        for (const auto &e : randomNums)
        {
            Mat<double> w_i                  = w.iloc(e, Axis::row);
            double      w_i_dot_theta        = w_i.dot(theta.transpose()).iloc(0, 0);
            double      exp_to_w_i_dot_theta = exp(w_i_dot_theta);
            double      p1                   = exp_to_w_i_dot_theta / (1 + exp_to_w_i_dot_theta);

            first_derivative -= (w_i * (mumerical_y.iloc(e, 0) - p1));
            second_derivative += (w_i.dot(w_i.transpose()) * p1 * (1 - p1)).iloc(0, 0);
        }

        theta -= learning_rate * first_derivative;
    }

    return theta;
}
Mat<std::string> LogisticRegression::predict_binary(const Mat<double> &x, const Mat<double> &theta) const
{
    using namespace std;

    Mat<double> probabilities = predict_probabilities(x, theta);
    Mat<string> ret(x.size(Axis::row), 1);
    for (size_t i = 0; i < x.size(Axis::row); ++i)
        ret.iloc(i, 0) = probabilities.iloc(i, 0) > 0.5 ? "T" : "F";
    return ret;
}
Mat<double> LogisticRegression::predict_probabilities(const Mat<double> &x, const Mat<double> &theta)
{
    using namespace std;

    if (x.size(Axis::col) + 1 != theta.size(Axis::col))
        throw invalid_argument("Error: Number of columns in x must be equal to number of columns in theta minus one.");

    Mat<double> y(x.size(Axis::row), 1);
    Mat<double> ones(x.size(Axis::row), 1);
    ones                    = 1;
    Mat<double> w           = x.concat(ones, Axis::col);
    double      w_dot_theta = w.dot(theta.transpose()).iloc(0, 0);
    for (size_t i = 0; i < x.size(Axis::row); ++i)
        y.iloc(i, 0) = 1 / (1 + exp(-w_dot_theta));

    return y;
}
std::shared_ptr<ClassificationModelBase<double>> LogisticRegression::clone() const
{
    using namespace std;

    return make_shared<LogisticRegression>(*this);
}