#include<random>
#include "ML/linearRegression.hpp"

using namespace TL;

// hook functions
Mat<double> LinearRegression::train_(const Mat<double> &x, const Mat<double> &y)
{
    using namespace std;

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

        Mat<double> tmp_theta(theta);

        for (size_t i = 0; i < w.size(Axis::col); ++i)
        {
            double tmp_theta_i = 0;
            for (auto &e : randomNums)
            {
                tmp_theta_i +=
                    learning_rate *
                    ((y.iloc(e, Axis::row) - theta.dot(w.iloc(e, Axis::row).transpose())) * w.iloc(e, i)).iloc(0, 0);
            }
            tmp_theta.iloc(0, i) += (tmp_theta_i / batch_size);
        }
        theta = tmp_theta;
    }
    return theta;
}
Mat<double> LinearRegression ::predict_(const Mat<double> &x, const Mat<double> &theta) const
{
    using namespace std;

    if (x.size(Axis::col) + 1 != theta.size(Axis::col))
    {
        cerr << "Error: The input matrix has incompatible dimensions with the model parameters." << endl;
        cerr << "Expected columns: " << (theta.size(Axis::col) - 1) << ", but got: " << x.size(Axis::col) << "."
             << endl;
        throw invalid_argument("The input matrix has incompatible dimensions with the model parameters.");
    }
    Mat<double> ones(x.size(Axis::row), 1);
    ones          = 1;
    Mat<double> w = x.concat(ones, Axis::col);
    return w.dot(theta.transpose());
}

// for polymorphism
std::shared_ptr<RegressionModelBase<double>> LinearRegression ::clone() const
{
    using namespace std;

    return make_shared<LinearRegression>(*this);
}