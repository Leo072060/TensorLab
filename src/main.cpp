#include <bitset>
#include <exception>
#include <filesystem>
#include <iostream>
#include <random>
#include <vector>

#include "ML/decisionTree.hpp"
#include "ML/evaluation.hpp"
#include "ML/linearModel.hpp"
#include "ML/multilayerPerceptron.hpp"
#include "ML/multilayerPerceptron_classification_b2m.hpp"
#include "mat/mat.hpp"
#include "preprocessor/split.hpp"
#include "utility/loader.hpp"
#include "utility/saver.hpp"

using namespace std;
using namespace TL;

#define TEST_MultilayerPerception_classification_b2m

int main()
{
    cout << "__main__" << endl;
    cout << "FLAG" << endl;
#ifdef TEST_mat
    csv_Loader<double> loader;

    string dataFileName = "matrix.csv";
    loader.nameFlag     = col_name;
    Mat<double> data    = loader.load_matrix(dataFileName);
    display_rainbow(data, col_name);
    auto origin_data = data;
    display_rainbow(origin_data);
    data = 7;
    display_rainbow(data, col_name);
    data += 10;
    display_rainbow(data, col_name);
    data -= 10;
    display_rainbow(data, col_name);
    data *= 10;
    display_rainbow(data, col_name);
    data /= 10;
    display_rainbow(data, col_name);
    data ^= 2;
    display_rainbow(data, col_name);
    cout << data.sum() << endl;
    display_rainbow(data.mean(Axis::row));
    display_rainbow(data.mean(Axis::col));
    display_rainbow(data.sum(Axis::row));
    display_rainbow(data.sum(Axis::col));
    display_rainbow(origin_data.inverse());
    auto LUP = origin_data.LU();
    display_rainbow(LUP["L"]);
    display_rainbow(LUP["U"]);
    display_rainbow(LUP["P"]);
    display(LUP["P"].dot(origin_data));
    display(LUP["L"].dot(LUP["U"]));
    cout << origin_data.det() << endl;
#endif

#ifdef TEST_LinearRegression
    csv_Loader loader;

    string dataFileName = "regression_data.csv";
    loader.nameFlag     = col_name;
    Mat data            = loader.load_matrix(dataFileName);
    display_rainbow(data, col_name, 5);

    LinearRegression model;

    auto x = data.extract(0, 5, Axis::col);
    auto y = data.loc("target", Axis::col);

    auto x_y = train_test_split(x, y, 0.2);

    auto x_train = x_y["x_train"];
    auto y_train = x_y["y_train"];
    auto x_test  = x_y["x_test"];
    auto y_test  = x_y["y_test"];

    model.train(x_train, y_train);
    display(model.get_theta());
    LinearRegression model_copy;
    model_copy  = model;
    auto y_pred = model_copy.predict(x_test);

    Evaluation_regression evalution;
    evalution.fit(y_pred, y_test);
    display_rainbow(y_pred.concat(y_test, Axis::col));
    evalution.report();
#endif

#ifdef TEST_LogisticRegression
    csv_Loader loader;
    string     dataFileName = "classification_data.csv";
    loader.nameFlag         = col_name;
    Mat data                = loader.load_matrix(dataFileName);
    display_rainbow(data, col_name, 5);
    auto        x = data.extract(0, data.size(Axis::col) - 1, Axis::col);
    Mat<string> y = data.loc("target", Axis::col);
    display(x);
    display(y);
    auto x_y = train_test_split(x, y, 0.2);

    auto x_train = x_y["x_train"];
    auto y_train = x_y["y_train"];
    auto x_test  = x_y["x_test"];
    auto y_test  = x_y["y_test"];

    LogisticRegression model;
    model.binary2multi = BinaryToMulti::OneVsOne;
    model.train(x_train, y_train);
    Evaluation_classification evalution;
    auto                      y_pred = model.predict(x_test);
    display_rainbow(y_test.concat(y_pred, Axis::col));
    evalution.fit(y_pred, y_test);
    evalution.report();
#endif

#ifdef TEST_DicisionTree
    csv_Loader<string> loader;
    string             dataFileName = "car_evaluation.csv";
    loader.nameFlag                 = col_name;
    auto data                       = loader.load_matrix(dataFileName);
    display_rainbow(data, col_name, 5);
    auto x = data.extract(0, data.size(Axis::col) - 1, Axis::col);
    auto y = data.loc("class", Axis::col);
    display_rainbow(x, NameFlag::col_name, 5);
    display_rainbow(y, NameFlag::col_name, 5);
    auto x_y = train_test_split(x, y, 0.2);

    auto x_train = x_y["x_train"];
    auto y_train = x_y["y_train"];
    auto x_test  = x_y["x_test"];
    auto y_test  = x_y["y_test"];

    DecisionTree model;
    model.train(x_train, y_train);

    Evaluation_classification evalution;
    auto                      y_pred = model.predict(x_test);
    display_rainbow(y_test.concat(y_pred, Axis::col));
    evalution.fit(y_pred, y_test);
    evalution.report();
#endif

#ifdef TEST_MultilayerPerception_regression
    csv_Loader loader;

    string dataFileName = "classification_data.csv";
    loader.nameFlag     = col_name;
    Mat data            = loader.load_matrix(dataFileName);
    display_rainbow(data, col_name, 5);
    cin.get();
    MultilayerPerception_regression model;

    data = data.extract(0, 500, Axis::row);

    auto x = data.extract(0, data.size(Axis::col) - 1, Axis::col);
    auto y = data.loc("target", Axis::col);

    auto x_y = train_test_split(x, y, 0.2);

    auto x_train = x_y["x_train"];
    auto y_train = x_y["y_train"];
    auto x_test  = x_y["x_test"];
    auto y_test  = x_y["y_test"];

    model.train(x_train, y_train);
    auto y_pred = model.predict(x_test);

    csv_Saver saver;
    saver.save_matrix(x_test.concat(y_test, Axis::col).concat(y_pred, Axis::col), "result.csv");

    Evaluation_regression evalution;
    evalution.fit(y_pred, y_test);
    display_rainbow(y_pred.concat(y_test, Axis::col));
    evalution.report();
#endif

#ifdef TEST_MultilayerPerception_classification
    csv_Loader loader;
    string     dataFileName = "classification_data.csv";
    loader.nameFlag         = col_name;
    Mat data                = loader.load_matrix(dataFileName);
    display_rainbow(data, col_name, 5);
    auto        x = data.extract(0, data.size(Axis::col) - 1, Axis::col);
    Mat<string> y = data.loc("target", Axis::col);
    display(x);
    display(y);
    auto x_y = train_test_split(x, y, 0.2);

    auto x_train = x_y["x_train"];
    auto y_train = x_y["y_train"];
    auto x_test  = x_y["x_test"];
    auto y_test  = x_y["y_test"];

    MultilayerPerception_classification model;
    model.train(x_train, y_train);
    Evaluation_classification evalution;
    auto                      y_pred = model.predict(x_test);
    display_rainbow(y_test.concat(y_pred, Axis::col));
    evalution.fit(y_pred, y_test);
    evalution.report();
#endif

#ifdef TEST_MultilayerPerception_classification_b2m
    csv_Loader loader;
    string     dataFileName = "classification_data.csv";
    loader.nameFlag         = col_name;
    Mat data                = loader.load_matrix(dataFileName);
    display_rainbow(data, col_name, 5);
    auto        x = data.extract(0, data.size(Axis::col) - 1, Axis::col);
    Mat<string> y = data.loc("target", Axis::col);
    display(x);
    display(y);
    auto x_y = train_test_split(x, y, 0.2);

    auto x_train = x_y["x_train"];
    auto y_train = x_y["y_train"];
    auto x_test  = x_y["x_test"];
    auto y_test  = x_y["y_test"];

    MultilayerPerception_classification_b2m model;
    model.train(x_train, y_train);
    Evaluation_classification evalution;
    auto                      y_pred = model.predict(x_test);
    display_rainbow(y_test.concat(y_pred, Axis::col));
    evalution.fit(y_pred, y_test);
    evalution.report();
#endif
}