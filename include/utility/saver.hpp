#ifndef DATA_SAVER_HPP
#define DATA_SAVER_HPP

#include <fstream>
#include <sstream>
#include <vector>

#include "mat/mat.hpp"

namespace TL
{
template <class T = double> class Saver
{
  protected:
    Saver() = default;

  public:
    virtual void save_matrix(const Mat<T> &mat, const std::string &fileName) const = 0;
};

template <class T = double> class csv_Saver : public Saver<T>
{
  public:
    void save_matrix(const Mat<T> &mat, const std::string &fileName) const override;

    NameFlag nameFlag = NameFlag::no_name;
};

template <typename T> void csv_Saver<T>::save_matrix(const Mat<T> &mat, const std::string &fileName) const
{
    using namespace std;

    ofstream file(fileName);
    if (!file.is_open())
    {
        cerr << "Error: Could not open file: " << fileName << endl;
        throw runtime_error("Could not open file: " + fileName);
    }

    // Write column names if they exist
    if (CheckFlag(nameFlag, col_name))
    {
        for (size_t c = 0; c < mat.size(Axis::col); ++c)
        {
            if (c > 0) file << ",";
            file << mat.iloc_name(c, Axis::col);
        }
        file << "\n";
    }

    // Write data rows
    for (size_t r = 0; r < mat.size(Axis::row); ++r)
    {
        // Write row name if it exists
        if (CheckFlag(nameFlag, row_name))
        {
            file << mat.iloc_name(r, Axis::row) << ",";
        }

        // Write row data
        for (size_t c = 0; c < mat.size(Axis::col); ++c)
        {
            if (c > 0) file << ",";
            file << mat.iloc(r, c);
        }
        file << "\n";
    }

    file.close();
}
} // namespace TL

#endif // DATA_SAVER_HPP
