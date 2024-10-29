#include <bitset>
#include <exception>
#include <filesystem>
#include <iostream>
#include <random>
#include <vector>

#include "ML/linearModel.hpp"
#include "kits/loader.hpp"
#include "mat/mat.hpp"

using namespace std;

#define TEST_linearModel
int main()
{
    cout << "__main__" << endl;

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

#ifdef TEST_linearModel
    csv_Loader loader;

    string dataFileName = "regression_data.csv";
    loader.nameFlag     = col_name;
    Mat data            = loader.load_matrix(dataFileName);
    display_rainbow(data, col_name, 5);

    LinearRegression model;

    auto x = data.extract(0, 5, Axis::col);
    auto y = data.loc("target", Axis::col);

    model.train(x, y);
    display(model.get_thetas());
    LinearRegression model_copy;
    model_copy = model;
    display_rainbow(model_copy.predict(x), no_name, 20);
#endif
}