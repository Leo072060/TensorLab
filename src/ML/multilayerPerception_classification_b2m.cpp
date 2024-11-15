#include <random>

#include "ML/multilayerPerceptron_classification_b2m.hpp"
#include "ML/multilayerPerceptron_regression.hpp"

using namespace TL;

Mat<double> MultilayerPerception_classification_b2m::train_binary(const Mat<double> &x, const Mat<std::string> &y)
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

    MultilayerPerception_regression mlp_regression;
    mlp_regression.iterations = 100000;

    mlp_regression.train(x, mumerical_y);

    return mlp_regression.get_theta();
}
Mat<std::string> MultilayerPerception_classification_b2m::predict_binary(const Mat<double> &x,
                                                                         const Mat<double> &theta) const
{
    using namespace std;

    MultilayerPerception_regression mlp_regression;
    mlp_regression.set_theta(theta);

    Mat<double> probabilities = mlp_regression.predict(x);

    Mat<string> ret(x.size(Axis::row), 1);
    for (size_t i = 0; i < x.size(Axis::row); ++i)
        ret.iloc(i, 0) = probabilities.iloc(i, 0) > 0.5 ? "T" : "F";

    return ret;
}
std::shared_ptr<ClassificationModelBase<double>> MultilayerPerception_classification_b2m::clone() const
{
    using namespace std;

    return make_shared<MultilayerPerception_classification_b2m>(*this);
}