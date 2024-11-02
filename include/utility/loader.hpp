#ifndef DATA_LOADER_HPP
#define DATA_LOADER_HPP

#include <fstream>
#include <sstream>
#include <vector>

#include "_internal/managed.hpp"
#include "mat/mat.hpp"

namespace TL
{
template <typename T> T str2T(const std::string &str)
{
    using namespace std;

    istringstream iss(str);
    T             val;
    iss >> val;
    if (iss.fail()) throw std::invalid_argument("Error: Invalid value: " + str);
    return val;
}

template <class T = double> class Loader
{
  protected:
    Loader() = default;

  public:
    virtual Mat<T> load_matrix(const std::string &fileName) const = 0;
};

template <class T = double> class csv_Loader : public Loader<T>
{
  public:
    Mat<T> load_matrix(const std::string &fileName) const override;

  public:
    NameFlag nameFlag = NameFlag::no_name;
};

template <typename T> Mat<T> csv_Loader<T>::load_matrix(const std::string &fileName) const
{
    using namespace std;
    
    ifstream file(fileName);
    if (!file.is_open())
    {
        cerr << "Error: Could not open file: " << fileName << endl;
        throw runtime_error("Could not open file: " + fileName);
    }

    vector<vector<T>> lines;
    string            line;
    vector<string>    colNames;
    vector<string>    rowNames;
    size_t            rowSize    = 0;
    size_t            columnSize = 0;

    // process the first row
    if (getline(file, line))
    {
        istringstream headerStream(line);
        string        headerCell;
        vector<T>     line_data;

        //  process the first cell
        getline(headerStream, headerCell, ',');
        // special handling for UTF-8 encoding to accommodate character data correctly
        if (headerCell.size() >= 3 && headerCell[0] == (char)0xEF && headerCell[1] == (char)0xBB &&
            headerCell[2] == (char)0xBF)
        {
            headerCell = headerCell.substr(3);
        }
        if (nameFlag == both_names)
            ;
        else if (nameFlag == col_name)
            colNames.push_back(headerCell);
        else if (nameFlag == row_name)
            rowNames.push_back(headerCell);
        else
            line_data.push_back(str2T<T>(headerCell));

        while (getline(headerStream, headerCell, ','))
        {
            if (CheckFlag(nameFlag, col_name))
                colNames.push_back(headerCell);
            else
                line_data.push_back(str2T<T>(headerCell));
        }
        if (!CheckFlag(nameFlag, col_name))
        {
            lines.push_back(line_data);
            ++rowSize;
        }
        columnSize = CheckFlag(nameFlag, col_name) ? colNames.size() : line_data.size();
    }
    else
        return Mat<T>();

    // process other rows
    while (getline(file, line))
    {
        istringstream ss(line);
        string        cell;
        vector<T>     line_data;
        getline(ss, cell, ',');
        if (CheckFlag(nameFlag, row_name))
            rowNames.push_back(cell);
        else
            line_data.push_back(str2T<T>(cell));
        while (getline(ss, cell, ','))
        {
            line_data.push_back(str2T<T>(cell));
        }
        if (line_data.size() != columnSize)
            throw runtime_error("Error: Row data size does not match the expected column size at row " +
                                to_string(rowSize + 1));
        lines.push_back(line_data);
        ++rowSize;
    }

    // store in the matrix
    Mat<T> mat(rowSize, columnSize);
    
    for (size_t i = 0; i < mat.size(Axis::row); ++i)
        for (size_t j = 0; j < mat.size(Axis::col); ++j)
            mat.iloc(i, j) = lines[i][j];
    if (CheckFlag(nameFlag, row_name))
        for (size_t i = 0; i < mat.size(Axis::row); ++i)
            mat.iloc_name(i, Axis::row) = rowNames[i];
    if (CheckFlag(nameFlag, col_name))
        for (size_t i = 0; i < mat.size(Axis::col); ++i)
            mat.iloc_name(i, Axis::col) = colNames[i];

    file.close();
    
    return mat;
}
} // namespace TL
#endif // DATA_LOADER_HPP