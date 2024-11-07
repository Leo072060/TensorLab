#ifndef MAT_HPP
#define MAT_HPP

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <unordered_map>

#include "_internal/managed.hpp"

namespace TL
{
using namespace _internal;

using NameFlagType = uint8_t;
enum NameFlag : uint8_t
{
    no_name    = 0b11111111,
    row_name   = 0b11111110,
    col_name   = 0b11111101,
    both_names = 0b11111100
};

enum class Order : int
{
    asce, // Ascending
    desc  // Descending
};

enum class Axis : int
{
    row,
    col,
    all
};

template <class T> class Mat;
// arithmetic operations
template <typename T> Mat<T> operator+(const T lhs, const Mat<T> &rhs);
template <typename T> Mat<T> operator-(const T lhs, const Mat<T> &rhs);
template <typename T> Mat<T> operator*(const T lhs, const Mat<T> &rhs);
template <typename T> Mat<T> operator/(const T lhs, const Mat<T> &rhs);

// display
template <typename T> void display(const Mat<T> &mat, const NameFlagType nameFlag = no_name, const size_t n_rows = -1);
template <typename T>
void display_rainbow(const Mat<T> &mat, const NameFlagType nameFlag = no_name, const size_t n_rows = -1);

template <class T = double> class Mat : public ManagedClass
{
    template <typename U> friend class Mat;

  public:
    // lifecycle management
    Mat();
    Mat(const Mat<T> &other);
    Mat(Mat<T> &&other) noexcept;
    Mat<T> &operator=(const Mat<T> &rhs);
    Mat<T> &operator=(Mat<T> &&rhs) noexcept;
    Mat(const size_t row, const size_t col);
    ~Mat();

    // type conversion
    template <typename U> operator Mat<U>() const;

    // metadata operations
    inline size_t size(const Axis axis = Axis::all) const;

    // arithmetic operations
    Mat<T> &operator=(const T rhs);
    Mat<T> &operator+=(const T rhs);
    Mat<T> &operator-=(const T rhs);
    Mat<T> &operator*=(const T rhs);
    Mat<T> &operator/=(const T rhs);
    Mat<T> &operator^=(const T rhs);
    Mat<T> &operator+=(const Mat<T> &rhs);
    Mat<T> &operator-=(const Mat<T> &rhs);
    Mat<T> &operator*=(const Mat<T> &rhs);
    Mat<T> &operator/=(const Mat<T> &rhs);

    Mat<T> operator+(const T rhs) const;
    Mat<T> operator-(const T rhs) const;
    Mat<T> operator*(const T rhs) const;
    Mat<T> operator/(const T rhs) const;
    Mat<T> operator^(const T rhs) const;
    bool   operator==(const Mat<T> &rhs) const;
    Mat<T> operator+(const Mat<T> &rhs) const;
    Mat<T> operator-(const Mat<T> &rhs) const;
    Mat<T> operator*(const Mat<T> &rhs) const;
    Mat<T> operator/(const Mat<T> &rhs) const;
    Mat<T> abs() const;
    Mat<T> dot(const Mat<T> &rhs) const;

    // evaluation operations
    T                                       mean() const;
    Mat<T>                                  mean(const Axis axis) const;
    T                                       sum() const;
    Mat<T>                                  sum(const Axis axis) const;
    Mat<T>                                  inverse() const;
    T                                       det() const;
    std::unordered_map<std::string, Mat<T>> LU() const;

    // indexing operations
    const T           &iloc(const size_t r, const size_t c) const;
    T                 &iloc(const size_t r, const size_t c);
    Mat<T>             iloc(const size_t i, const Axis axis) const;
    const T           &loc(const std::string &rowName, const std::string &colName) const;
    T                 &loc(const std::string &rowName, const std::string &colName);
    Mat<T>             loc(const std::string &name, const Axis axis) const;
    const std::string &iloc_name(const size_t i, const Axis axis) const;
    std::string       &iloc_name(const size_t i, const Axis axis);

    Mat<T> extract(const size_t row_begin, const size_t row_end, const size_t col_begin, const size_t col_end) const;
    Mat<T> extract(const size_t begin, const size_t end, const Axis axis) const;
    Mat<T> extract(std::vector<size_t> index, Axis axis) const;

    // matrix operations
    Mat<T> concat(const Mat<T> &other, const Axis axis) const;
    Mat<T> transpose() const;
    void   swap(const size_t a, const size_t b, const Axis axis);
    void   drop(const size_t begin, const size_t end, const Axis axis);
    void   drop(const size_t i, const Axis axis);

    // algorithm operation
    void                                   sort(const size_t i, const Order order, const Axis axis);
    std::vector<std::pair<size_t, size_t>> find(const T val) const;
    size_t                                 count(const T val) const;
    std::unordered_map<T, size_t>          count() const;
    Mat<T>                                 unique() const;

    // matrix generation
    static Mat<T> identity(const size_t size);

    // tool functions
    size_t name2index(const std::string &name, const Axis axis) const;

  private:
    T          **data     = nullptr;
    std::string *rowNames = nullptr;
    std::string *colNames = nullptr;
    std::size_t  rowSize  = 0;
    std::size_t  colSize  = 0;

    // calculated value
    mutable ManagedVal<T> managed_det;
    mutable ManagedVal<T> managed_mean;
    mutable ManagedVal<T> managed_sum;
};

// lifecycle management
template <typename T>
Mat<T>::Mat()
    : ManagedClass()
    , managed_det(this->administrator)
    , managed_mean(this->administrator)
    , managed_sum(this->administrator)
{
}
template <typename T>
Mat<T>::Mat(const Mat<T> &other)
    : ManagedClass(other)
    , managed_det(this->administrator, other.administrator, other.managed_det)
    , managed_mean(this->administrator, other.administrator, other.managed_mean)
    , managed_sum(this->administrator, other.administrator, other.managed_sum)
{
    using namespace std;

    rowSize = other.rowSize;
    colSize = other.colSize;

    data     = new T *[rowSize];
    rowNames = new string[rowSize];
    colNames = new string[colSize];

    for (size_t r = 0; r < rowSize; ++r)
    {
        data[r]     = new T[colSize];
        rowNames[r] = other.rowNames[r];
        for (size_t c = 0; c < colSize; ++c)
            data[r][c] = other.data[r][c];
    }

    for (size_t c = 0; c < colSize; ++c)
        colNames[c] = other.colNames[c];
}
template <typename T>
Mat<T>::Mat(Mat<T> &&other) noexcept
    : ManagedClass(std::move(other))
    , managed_det(this->administrator, other.administrator, other.managed_det)
    , managed_mean(this->administrator, other.administrator, other.managed_mean)
    , managed_sum(this->administrator, other.administrator, other.managed_sum)
{
    data     = other.data;
    rowNames = other.rowNames;
    colNames = other.colNames;
    rowSize  = other.rowSize;
    colSize  = other.colSize;

    other.data     = nullptr;
    other.rowNames = nullptr;
    other.colNames = nullptr;
    other.rowSize  = 0;
    other.colSize  = 0;
}
template <typename T> Mat<T> &Mat<T>::operator=(const Mat<T> &rhs)
{
    using namespace std;

    ManagedClass::operator=(rhs);
    managed_det.copy(this->administrator, rhs.administrator, rhs.managed_det);
    managed_mean.copy(this->administrator, rhs.administrator, rhs.managed_mean);
    managed_sum.copy(this->administrator, rhs.administrator, rhs.managed_sum);

    if (this == &rhs) return *this;

    for (size_t r = 0; r < rowSize; ++r)
        delete[] data[r];
    delete[] rowNames;
    delete[] colNames;

    data = new T *[rhs.rowSize];
    for (size_t r = 0; r < rhs.rowSize; ++r)
    {
        data[r] = new T[rhs.colSize];
        for (size_t c = 0; c < rhs.colSize; ++c)
            data[r][c] = rhs.data[r][c];
    }
    rowNames = new string[rhs.rowSize];
    for (size_t r = 0; r < rhs.rowSize; ++r)
        rowNames[r] = rhs.rowNames[r];
    colNames = new string[rhs.colSize];
    for (size_t c = 0; c < rhs.colSize; ++c)
        colNames[c] = rhs.colNames[c];

    rowSize = rhs.rowSize;
    colSize = rhs.colSize;

    return *this;
}
template <typename T> Mat<T> &Mat<T>::operator=(Mat<T> &&rhs) noexcept
{
    using namespace std;

    ManagedClass::operator=(move(rhs));
    managed_det.copy(this->administrator, rhs.administrator, rhs.managed_det);
    managed_mean.copy(this->administrator, rhs.administrator, rhs.managed_mean);
    managed_sum.copy(this->administrator, rhs.administrator, rhs.managed_sum);

    if (this == &rhs) return *this;

    for (size_t r = 0; r < rowSize; ++r)
        delete[] data[r];
    delete[] rowNames;
    delete[] colNames;

    data     = rhs.data;
    rowNames = rhs.rowNames;
    colNames = rhs.colNames;
    rowSize  = rhs.rowSize;
    colSize  = rhs.colSize;

    rhs.data     = nullptr;
    rhs.rowNames = nullptr;
    rhs.colNames = nullptr;
    rhs.rowSize  = 0;
    rhs.colSize  = 0;

    return *this;
}
template <typename T>
Mat<T>::Mat(const size_t row, const size_t col)
    : Mat()
{
    using namespace std;

    data     = new T *[row];
    rowNames = new string[row];
    colNames = new string[col];
    for (size_t r = 0; r < row; ++r)
    {
        data[r] = new T[col];
        for (size_t c = 0; c < col; ++c)
            data[r][c] = T();
    }

    this->rowSize = row;
    this->colSize = col;
}
template <typename T> Mat<T>::~Mat()
{
    if (data)
        for (size_t r = 0; r < rowSize; ++r)
            delete[] data[r];
    if (rowNames) delete[] rowNames;
    if (colNames) delete[] colNames;

    data     = nullptr;
    rowNames = nullptr;
    colNames = nullptr;
}

// type conversion
template <typename T> template <typename U> Mat<T>::operator Mat<U>() const
{
    using namespace std;

    Mat<U> ret(rowSize, colSize);

    for (size_t r = 0; r < rowSize; ++r)
        ret.rowNames[r] = rowNames[r];
    for (size_t c = 0; c < colSize; ++c)
        ret.colNames[c] = colNames[c];
    for (size_t r = 0; r < rowSize; ++r)
        for (size_t c = 0; c < colSize; ++c)
        {
            stringstream ss;
            ss << data[r][c];
            U value;
            ss >> value;
            ret.data[r][c] = value;
        }

    return ret;
}

// metadata operations
template <typename T> size_t Mat<T>::size(const Axis axis) const
{
    using namespace std;

    switch (axis)
    {
    case Axis::row:
        return rowSize;
    case Axis::col:
        return colSize;
    case Axis::all:
        return rowSize * colSize;
    default:
        cerr << "Error: Invalid Axis provided. " << endl;
        throw invalid_argument("Invalid Axis.");
    }
}

// arithmetic operations
template <typename T> Mat<T> &Mat<T>::operator=(const T rhs)
{
    for (size_t r = 0; r < rowSize; ++r)
        for (size_t c = 0; c < colSize; ++c)
            data[r][c] = rhs;
    return *this;
}
template <typename T> Mat<T> &Mat<T>::operator+=(const T rhs)
{
    for (size_t r = 0; r < rowSize; ++r)
        for (size_t c = 0; c < colSize; ++c)
            data[r][c] += rhs;
    return *this;
}
template <typename T> Mat<T> &Mat<T>::operator-=(const T rhs)
{
    for (size_t r = 0; r < rowSize; ++r)
        for (size_t c = 0; c < colSize; ++c)
            data[r][c] -= rhs;
    return *this;
}
template <typename T> Mat<T> &Mat<T>::operator*=(const T rhs)
{
    for (size_t r = 0; r < rowSize; ++r)
        for (size_t c = 0; c < colSize; ++c)
            data[r][c] *= rhs;
    return *this;
}
template <typename T> Mat<T> &Mat<T>::operator/=(const T rhs)
{
    for (size_t r = 0; r < rowSize; ++r)
        for (size_t c = 0; c < colSize; ++c)
            data[r][c] /= rhs;
    return *this;
}
template <typename T> Mat<T> &Mat<T>::operator^=(const T rhs)
{
    using namespace std;

    for (size_t r = 0; r < rowSize; ++r)
        for (size_t c = 0; c < colSize; ++c)
            data[r][c] = pow(data[r][c], rhs);

    return *this;
}
template <typename T> Mat<T> &Mat<T>::operator+=(const Mat<T> &rhs)
{
    using namespace std;

    if (rowSize != rhs.rowSize || colSize != rhs.colSize)
    {
        cerr << "Error: Cannot perform addition. Matrix dimension mismatch: "
             << "Left-hand side matrix dimensions "
             << "(rows: " << rowSize << ", cols: " << colSize << ")"
             << "do not match right-hand side matrix dimensions "
             << "(rows: " << rhs.rowSize << ", cols: " << rhs.colSize << ")"
             << "." << endl;
        throw invalid_argument("Matrix dimensions must match for addition operation.");
    }

    for (size_t r = 0; r < rowSize; ++r)
        for (size_t c = 0; c < colSize; ++c)
            data[r][c] += rhs.data[r][c];

    return *this;
}
template <typename T> Mat<T> &Mat<T>::operator-=(const Mat<T> &rhs)
{
    using namespace std;

    if (rowSize != rhs.rowSize || colSize != rhs.colSize)
    {
        cerr << "Error: Cannot perform subtraction. Matrix dimension mismatch:  "
             << "Left-hand side matrix dimensions "
             << "(rows: " << rowSize << ", cols: " << colSize << ")"
             << "do not match right-hand side matrix dimensions "
             << "(rows: " << rhs.rowSize << ", cols: " << rhs.colSize << ")"
             << "." << endl;
        throw invalid_argument("Matrix dimensions must match for subtraction operation.");
    }

    for (size_t r = 0; r < rowSize; ++r)
        for (size_t c = 0; c < colSize; ++c)
            data[r][c] -= rhs.data[r][c];

    return *this;
}
template <typename T> Mat<T> &Mat<T>::operator*=(const Mat<T> &rhs)
{
    using namespace std;

    if (rowSize != rhs.rowSize || colSize != rhs.colSize)
    {
        cerr << "Error: Cannot perform multiplication. Matrix dimension mismatch:  "
             << "Left-hand side matrix dimensions "
             << "(rows: " << rowSize << ", cols: " << colSize << ")"
             << "do not match right-hand side matrix dimensions "
             << "(rows: " << rhs.rowSize << ", cols: " << rhs.colSize << ")"
             << "." << endl;
        throw invalid_argument("Matrix dimensions must match for multiplication operation.");
    }

    for (size_t r = 0; r < rowSize; ++r)
        for (size_t c = 0; c < colSize; ++c)
            data[r][c] *= rhs.data[r][c];

    return *this;
}
template <typename T> Mat<T> &Mat<T>::operator/=(const Mat<T> &rhs)
{
    using namespace std;

    if (rowSize != rhs.rowSize || colSize != rhs.colSize)
    {
        cerr << "Error: Cannot perform division. Matrix dimension mismatch:  "
             << "Left-hand side matrix dimensions "
             << "(rows: " << rowSize << ", cols: " << colSize << ")"
             << "do not match right-hand side matrix dimensions "
             << "(rows: " << rhs.rowSize << ", cols: " << rhs.colSize << ")"
             << "." << endl;
        throw invalid_argument("Matrix dimensions must match for division operation.");
    }

    for (size_t r = 0; r < rowSize; ++r)
        for (size_t c = 0; c < colSize; ++c)
            data[r][c] /= rhs.data[r][c];

    return *this;
}
template <typename T> Mat<T> Mat<T>::operator+(const T rhs) const
{
    Mat<T> ret(*this);
    return ret += rhs;
}
template <typename T> Mat<T> Mat<T>::operator-(const T rhs) const
{
    Mat<T> ret(*this);
    return ret -= rhs;
}
template <typename T> Mat<T> Mat<T>::operator*(const T rhs) const
{
    Mat<T> ret(*this);
    return ret *= rhs;
}
template <typename T> Mat<T> Mat<T>::operator/(const T rhs) const
{
    Mat<T> ret(*this);
    return ret /= rhs;
}
template <typename T> Mat<T> Mat<T>::operator^(const T rhs) const
{
    Mat<T> ret(*this);
    return ret ^= rhs;
}
template <typename T> bool Mat<T>::operator==(const Mat<T> &rhs) const
{
    if (rowSize != rhs.rowSize || colSize != rhs.colSize) return false;
    for (size_t r = 0; r < rowSize; ++r)
        for (size_t c = 0; c < colSize; ++c)
            if (data[r][c] != rhs.data[r][c]) return false;
    return true;
}
template <typename T> Mat<T> Mat<T>::operator+(const Mat<T> &rhs) const
{
    Mat<T> ret(*this);
    return ret += rhs;
}
template <typename T> Mat<T> Mat<T>::operator-(const Mat<T> &rhs) const
{
    Mat<T> ret(*this);
    return ret -= rhs;
}
template <typename T> Mat<T> Mat<T>::operator*(const Mat<T> &rhs) const
{
    Mat<T> ret(*this);
    return ret *= rhs;
}
template <typename T> Mat<T> Mat<T>::operator/(const Mat<T> &rhs) const
{
    Mat<T> ret(*this);
    return ret /= rhs;
}
template <typename T> Mat<T> Mat<T>::abs() const
{
    Mat<T> ret(*this);
    for (size_t r = 0; r < rowSize; ++r)
        for (size_t c = 0; c < colSize; ++c)
            ret.data[r][c] = std::abs(ret.data[r][c]);
    return ret;
}
template <typename T> Mat<T> Mat<T>::dot(const Mat<T> &rhs) const
{
    using namespace std;

    if (colSize != rhs.rowSize)
    {
        cerr << "Error: Cannot perform dot product. Matrix dimension mismatch: "
             << "Left-hand side matrix dimensions (cols: " << colSize << ")"
             << " do not align with right-hand side matrix dimensions (rows: " << rhs.rowSize << ")." << endl;
        throw invalid_argument("Matrix dimensions do not align for dot product operation.");
    }

    Mat<T> ret(rowSize, rhs.colSize);
    for (size_t r = 0; r < rowSize; ++r)
        for (size_t c = 0; c < rhs.colSize; ++c)
            for (size_t k = 0; k < colSize; ++k)
                ret.data[r][c] += data[r][k] * rhs.data[k][c];

    return ret;
}
template <typename T> Mat<T> operator+(const T lhs, const Mat<T> &rhs)
{
    return rhs + lhs;
}
template <typename T> Mat<T> operator-(const T lhs, const Mat<T> &rhs)
{
    return rhs - lhs;
}
template <typename T> Mat<T> operator*(const T lhs, const Mat<T> &rhs)
{
    return rhs * lhs;
}
template <typename T> Mat<T> operator/(const T lhs, const Mat<T> &rhs)
{
    return rhs / lhs;
}

// evaluation operations
template <typename T> T Mat<T>::mean() const
{
    return mean(Axis::all).data[0][0];
}
template <typename T> Mat<T> Mat<T>::mean(const Axis axis) const
{
    using namespace std;

    if (0 == size())
    {
        cerr << "Error: Cannot compute mean of an empty matrix." << endl;
        throw invalid_argument("Matrix is empty.");
    }

    Mat<T> ret(sum(axis));
    switch (axis)
    {
    case Axis::row:
        for (size_t r = 0; r < rowSize; ++r)
            ret.data[r][0] /= colSize;
        break;
    case Axis::col:
        for (size_t c = 0; c < colSize; ++c)
            ret.data[0][c] /= rowSize;
        break;
    case Axis::all:
        ret.data[0][0] /= size();
        break;
    default:
        cerr << "Error: Invalid Axis provided for mean operation. " << endl;
        throw invalid_argument("Invalid Axis for mean operation.");
    }

    return ret;
}
template <typename T> T Mat<T>::sum() const
{
    return sum(Axis::all).data[0][0];
}
template <typename T> Mat<T> Mat<T>::sum(const Axis axis) const
{
    using namespace std;

    if (0 == size())
    {
        cerr << "Error: Cannot compute sum of an empty matrix." << endl;
        throw invalid_argument("Matrix is empty.");
    }

    // Use Kahan compensation to improve accuracy of the sum calculation
    switch (axis)
    {
    case Axis::row: {
        Mat<T> ret(rowSize, 1);
        for (size_t r = 0; r < rowSize; ++r)
        {
            T sum  = 0; // cumulative sum for the current row
            T comp = 0; // compensation for lost low-order bits
            for (size_t c = 0; c < colSize; ++c)
            {
                T y  = data[r][c] - comp; // adjust for compensation
                T t  = sum + y;           // combine the current sum and the adjusted value
                comp = (t - sum) - y;     // update compensation for lost low-order bits
                sum  = t;                 // update the cumulative sum
            }
            ret.data[r][0] = sum;
        }
        return ret;
    }
    case Axis::col: {
        Mat<T> ret(1, colSize);
        for (size_t c = 0; c < colSize; ++c)
        {
            T sum  = 0;
            T comp = 0;
            for (size_t r = 0; r < rowSize; ++r)
            {
                T y  = data[r][c] - comp;
                T t  = sum + y;
                comp = (t - sum) - y;
                sum  = t;
            }
            ret.data[0][c] = sum;
        }
        return ret;
    }
    case Axis::all: {
        Mat<T> ret(1, 1);
        T      sum  = 0;
        T      comp = 0;
        for (size_t r = 0; r < rowSize; ++r)
        {
            for (size_t c = 0; c < colSize; ++c)
            {
                T y  = data[r][c] - comp;
                T t  = sum + y;
                comp = (t - sum) - y;
                sum  = t;
            }
        }
        ret.data[0][0] = sum;
        return ret;
    }
    default:
        cerr << "Error: Invalid Axis provided for sum operation. " << endl;
        throw invalid_argument("Invalid Axis for sum operation.");
    }
}
template <typename T> Mat<T> Mat<T>::inverse() const
{
    using namespace std;

    if (rowSize != colSize)
    {
        cerr << "Error: Matrix must be square to compute inverse. Given dimensions: " << rowSize << "x" << colSize
             << "." << endl;
        throw invalid_argument("Matrix must be square to compute inverse.");
    }

    Mat<T> A(*this);
    Mat<T> I = identity(rowSize);

    for (size_t d = 0; d < rowSize; ++d) // d is for depth
    {
        // partial pivoting
        T      maxVal    = fabs(A.data[d][d]);
        size_t maxValRow = d;
        for (size_t r = d + 1; r < rowSize; ++r)
        {
            if (fabs(A.data[r][d]) > maxVal)
            {
                maxVal    = fabs(A.data[r][d]);
                maxValRow = r;
            }
        }
        if (maxValRow != d)
        {
            A.swap(d, maxValRow, Axis::row);
            I.swap(d, maxValRow, Axis::row);
        }
        if (fabs(A.data[d][d]) < 1e-9)
        {
            cerr << "Error: Matrix is singular at index " << d << " and cannot be inverted." << endl;
            throw runtime_error("Matrix is singular and cannot be inverted.");
        }

        T pivot = A.data[d][d];
        for (size_t c = 0; c < rowSize; ++c)
        {
            A.data[d][c] /= pivot;
            I.data[d][c] /= pivot;
        }
        for (size_t r = 0; r < rowSize; ++r)
        {
            if (r != d)
            {
                T factor = A.data[r][d];
                for (size_t c = 0; c < rowSize; ++c)
                {
                    A.data[r][c] -= factor * A.data[d][c];
                    I.data[r][c] -= factor * I.data[d][c];
                }
            }
        }
    }

    return I;
}
template <typename T> T Mat<T>::det() const
{
    using namespace std;

    if (managed_det.isReadable()) return managed_det.read();

    if (0 == size())
    {
        cerr << "Error: Cannot compute the determinant of an empty matrix." << endl;
        throw invalid_argument("Matrix is empty.");
    }
    if (rowSize != colSize)
    {
        cerr << "Error: Determinant requires a square matrix. Given matrix dimensions: " << rowSize << "x" << colSize
             << "." << endl;
        throw invalid_argument("Determinant requires a square matrix.");
    }

    map<string, Mat<T>> LU_decomp = LU();
    Mat<T>              U         = LU_decomp["U"];
    Mat<T>              P         = LU_decomp["P"];
    T                   ret       = 1;
    for (size_t i = 0; i < rowSize; ++i)
        ret *= U.data[i][i];

    // calculate the number of row swaps in the permutation matrix P
    size_t         swapCount = 0;
    vector<size_t> location(rowSize);
    for (size_t r = 0; r < rowSize; ++r)
        for (size_t c = 0; c < colSize; ++c)
            if (P.data[r][c] == 1)
            {
                location[r] = c;
                break;
            }

    for (size_t i = 0; i < rowSize; ++i)
        for (size_t j = i + 1; j < rowSize; ++j)
            if (location[i] > location[j])
            {
                swapCount++;
            }

    ret *= (swapCount % 2 == 0) ? 1 : -1; // multiply det by -1 when P has odd swaps

    this->record(managed_det, ret);
    return ret;
}
template <typename T> std::unordered_map<std::string, Mat<T>> Mat<T>::LU() const
{
    using namespace std;

    if (rowSize != colSize)
    {
        cerr << "Error: LU decomposition requires a square matrix. Given matrix dimensions: " << rowSize << "x"
             << colSize << "." << endl;
        throw invalid_argument("LU decomposition requires a square matrix.");
    }

    Mat<T> L(rowSize, rowSize);
    Mat<T> U = *this;
    Mat<T> P = identity(rowSize);
    for (size_t r = 0; r < rowSize; ++r)
    {
        size_t pivot = r;
        for (size_t c = r + 1; c < rowSize; ++c)
            if (std::abs(U.data[c][r]) > std::abs(U.data[pivot][r])) pivot = c;
        if (pivot != r)
        {
            U.swap(r, pivot, Axis::row);
            Mat<T> I = identity(rowSize);
            P.swap(r, pivot, Axis::row);
            L.swap(r, pivot, Axis::row);
        }
        if (U.data[r][r] == 0)
        {
            cerr << "Error: Matrix is singular at index " << r << " and cannot be decomposed." << endl;
            throw runtime_error("Matrix is singular and cannot be decomposed.");
        }

        for (size_t c = r + 1; c < rowSize; ++c)
        {
            T factor     = U.data[c][r] / U.data[r][r];
            L.data[c][r] = factor;
            for (size_t k = r; k < rowSize; ++k)
                U.data[c][k] -= factor * U.data[r][k];
        }
    }

    L += Mat<T>::identity(rowSize);

    unordered_map<string, Mat<T>> ret;
    ret["L"] = L;
    ret["U"] = U;
    ret["P"] = P;
    return ret;
}

// indexing operations
template <typename T> T &Mat<T>::iloc(const size_t r, const size_t c)
{
    this->refresh();
    return data[r][c];
}
template <typename T> const T &Mat<T>::iloc(const size_t r, const size_t c) const
{
    return data[r][c];
}
template <typename T> Mat<T> Mat<T>::iloc(const size_t i, const Axis axis) const
{
    using namespace std;

    switch (axis)
    {
    case Axis::row: {
        if (i >= rowSize)
        {
            cerr << "Error: Row index " << i << " is out of bounds. Matrix has " << rowSize << " rows." << endl;
            throw out_of_range("Row index is out of bounds.");
        }
        Mat<T> ret(1, colSize);
        for (size_t c = 0; c < colSize; ++c)
        {
            ret.data[0][c]  = data[i][c];
            ret.colNames[c] = rowNames[c];
        }
        ret.rowNames[0] = rowNames[i];
        return ret;
    }
    case Axis::col: {
        if (i >= colSize)
        {
            cerr << "Error: Column index " << i << " is out of bounds. Matrix has " << colSize << " columns." << endl;
            throw out_of_range("Column index is out of bounds.");
        }
        Mat<T> ret(rowSize, 1);
        for (size_t r = 0; r < rowSize; ++r)
        {
            ret.data[r][0]  = data[r][i];
            ret.rowNames[r] = rowNames[r];
        }
        ret.colNames[0] = colNames[i];
        return ret;
    }
    default:
        cerr << "Error: Invalid Axis provided for iloc operation. " << endl;
        throw invalid_argument("Invalid Axis for iloc operation.");
    }
}
template <typename T> const T &Mat<T>::loc(const std::string &rowName, const std::string &colName) const
{
    using namespace std;

    size_t r = 0, c = 0;
    for (; r < rowSize; ++r)
        if (rowNames[r] == rowName) break;
    if (r == rowSize)
    {
        cerr << "Error: Row name '" << rowName << "' not found." << endl;
        throw invalid_argument("Row name not found.");
    }
    for (; c < colSize; ++c)
        if (colNames[c] == colName) break;
    if (c == colSize)
    {
        cerr << "Error: Column name '" << colName << "' not found." << endl;
        throw invalid_argument("Column name not found.");
    }

    return data[r][c];
}
template <typename T> T &Mat<T>::loc(const std::string &rowName, const std::string &colName)
{
    this->refresh();
    return const_cast<T &>(static_cast<const Mat<T>>(*this).loc(rowName, colName));
}
template <typename T> Mat<T> Mat<T>::loc(const std::string &name, const Axis axis) const
{
    using namespace std;

    switch (axis)
    {
    case Axis::row: {
        size_t r = 0;
        for (; r < rowSize; ++r)
            if (rowNames[r] == name) break;
        if (r == rowSize)
        {
            cerr << "Error: Row name '" << name << "' not found." << endl;
            throw invalid_argument("Row name not found.");
        }
        Mat<T> ret(1, colSize);
        for (size_t c = 0; c < colSize; ++c)
        {
            ret.data[0][c]  = data[r][c];
            ret.colNames[c] = colNames[c];
        }
        ret.rowNames[0] = rowNames[r];
        return ret;
    }
    case Axis::col: {
        size_t c = 0;
        for (; c < colSize; ++c)
            if (colNames[c] == name) break;
        if (c == colSize)
        {
            cerr << "Error: Column name '" << name << "' not found." << endl;
            throw invalid_argument("Column name not found.");
        }
        Mat<T> ret(rowSize, 1);
        for (size_t r = 0; r < rowSize; ++r)
        {
            ret.data[r][0]  = data[r][c];
            ret.rowNames[r] = rowNames[r];
        }
        ret.colNames[0] = colNames[c];
        return ret;
    }
    default:
        cerr << "Error: Invalid Axis provided for loc operation. " << endl;
        throw invalid_argument("Invalid Axis for loc operation.");
    }
}
template <typename T> const std::string &Mat<T>::iloc_name(const size_t i, const Axis axis) const
{
    using namespace std;

    switch (axis)
    {
    case Axis::row:
        if (i >= rowSize)
        {
            cerr << "Error: Row index out of bounds. Provided index: " << i << ", but rowSize = " << rowSize << "."
                 << endl;
            throw out_of_range("Row index out of bounds.");
        }
        return rowNames[i];
    case Axis::col:
        if (i >= colSize)
        {
            cerr << "Error: Column index out of bounds. Provided index: " << i << ", but colSize = " << colSize << "."
                 << endl;
            throw out_of_range("Column index out of bounds.");
        }
        return colNames[i];
    default:
        cerr << "Error: Invalid Axis provided for iloc_name operation. " << endl;
        throw invalid_argument("Invalid Axis for iloc_name operation.");
    }
}
template <typename T> std::string &Mat<T>::iloc_name(const size_t i, const Axis axis)
{
    using namespace std;
    this->refresh();
    return const_cast<string &>(static_cast<const Mat<T> &>(*this).iloc_name(i, axis));
}
template <typename T>
Mat<T> Mat<T>::extract(const size_t row_begin, const size_t row_end, const size_t col_begin, const size_t col_end) const
{
    using namespace std;

    if (row_begin >= rowSize || row_end > rowSize || col_begin >= colSize || col_end > colSize)
    {
        cerr << "Error: Indices are out of bounds." << endl;
        throw out_of_range("Index out of bounds.");
    }
    if (row_begin >= row_end || col_begin >= col_end)
    {
        cerr << "Error: Invalid extraction boundaries." << endl;
        throw invalid_argument("Invalid extraction boundaries.");
    }

    Mat<T> ret(row_end - row_begin, col_end - col_begin);
    for (size_t r = row_begin; r < row_end; ++r)
    {
        ret.rowNames[r - row_begin] = rowNames[r];
        for (size_t c = col_begin; c < col_end; ++c)
            ret.data[r - row_begin][c - col_begin] = data[r][c];
    }
    for (size_t c = col_begin; c < col_end; ++c)
        ret.colNames[c - col_begin] = colNames[c];

    return ret;
}
template <typename T> Mat<T> Mat<T>::extract(const size_t begin, const size_t end, const Axis axis) const
{
    using namespace std;

    if (begin >= end)
    {
        cerr << "Error: Invalid extraction boundaries." << endl;
        throw invalid_argument("Invalid extraction boundaries.");
    }

    switch (axis)
    {
    case Axis::row: {
        if (end > rowSize)
        {
            cerr << "Error: Row indices are out of bounds." << endl;
            throw out_of_range("Row index out of bounds.");
        }

        Mat<T> ret(end - begin, colSize);
        for (size_t r = begin; r < end; ++r)
        {
            ret.rowNames[r - begin] = rowNames[r];
            for (size_t c = 0; c < colSize; ++c)
                ret.data[r - begin][c] = data[r][c];
        }
        for (size_t c = 0; c < colSize; ++c)
            ret.colNames[c] = colNames[c];
        return ret;
    }
    case Axis::col: {
        if (end > colSize)
        {
            cerr << "Error: Column indices are out of bounds." << endl;
            throw out_of_range("Column index out of bounds.");
        }

        Mat<T> ret(rowSize, end - begin);
        for (size_t c = begin; c < end; ++c)
        {
            ret.colNames[c - begin] = colNames[c];
            for (size_t r = 0; r < rowSize; ++r)
            {
                ret.data[r][c - begin] = data[r][c];
            }
        }
        for (size_t r = 0; r < rowSize; ++r)
            ret.rowNames[r] = rowNames[r];

        return ret;
    }
    default:
        cerr << "Error: Invalid Axis provided for extraction operation. " << endl;
        throw invalid_argument("Invalid Axis for extraction operation.");
    }
}
template <typename T> Mat<T> Mat<T>::extract(std::vector<size_t> index, Axis axis) const
{
    using namespace std;

    switch (axis)
    {
    case Axis::row: {
        for (size_t e : index)
            if (e >= rowSize)
            {
                cerr << "Error: Row index " << e << " is out of bounds." << endl;
                throw out_of_range("Row index out of bounds.");
            }

        Mat<T> ret(index.size(), colSize);
        for (size_t r = 0; r < index.size(); ++r)
        {
            ret.rowNames[r] = rowNames[index[r]];
            for (size_t c = 0; c < colSize; ++c)
                ret.data[r][c] = data[index[r]][c];
        }
        for (size_t c = 0; c < colSize; ++c)
            ret.colNames[c] = colNames[c];

        return ret;
    }
    case Axis::col: {
        for (size_t e : index)
            if (e >= colSize)
            {
                cerr << "Error: Column index " << e << " is out of bounds." << endl;
                throw out_of_range("Column index out of bounds.");
            }

        Mat<T> ret(rowSize, index.size());
        for (size_t c = 0; c < index.size(); ++c)
        {
            ret.colNames[c] = colNames[index[c]];
            for (size_t r = 0; r < rowSize; ++r)
                ret.data[r][c] = data[r][index[c]];
        }
        for (size_t r = 0; r < rowSize; ++r)
            ret.rowNames[r] = rowNames[r];

        return ret;
    }
    default:
        cerr << "Error: Invalid Axis provided for extraction operation. " << endl;
        throw invalid_argument("Invalid Axis for extraction operation.");
    }
}

// matrix operations
template <typename T> Mat<T> Mat<T>::concat(const Mat<T> &other, const Axis axis) const
{
    using namespace std;

    switch (axis)
    {
    case Axis::row: {
        if (rowSize && other.rowSize && colSize != other.colSize)
        {
            cerr << "Error: Column sizes must match for row concatenation." << endl;
            throw invalid_argument("Column sizes do not match for row concatenation.");
        }

        Mat<T> ret(rowSize + other.rowSize, colSize ? colSize : other.colSize);
        for (size_t r = 0; r < rowSize; ++r)
        {
            ret.rowNames[r] = rowNames[r];
            for (size_t c = 0; c < colSize; ++c)
                ret.data[r][c] = data[r][c];
        }
        for (size_t r = 0; r < other.rowSize; ++r)
        {
            ret.rowNames[r + rowSize] = other.rowNames[r];
            for (size_t c = 0; c < other.colSize; ++c)
                ret.data[r + rowSize][c] = other.data[r][c];
        }
        if (0 == size())
            for (size_t c = 0; c < other.colSize; ++c)
                ret.colNames[c] = other.colNames[c];
        else
            for (size_t c = 0; c < colSize; ++c)
                ret.colNames[c] = colNames[c];

        return ret;
    }
    case Axis::col: {
        if (colSize && other.colSize && rowSize != other.rowSize)
        {
            cerr << "Error: Row sizes must match for column concatenation." << endl;
            throw invalid_argument("Row sizes do not match for column concatenation.");
        }

        Mat<T> ret(rowSize ? rowSize : other.rowSize, colSize + other.colSize);
        for (size_t c = 0; c < colSize; ++c)
        {
            ret.colNames[c] = colNames[c];
            for (size_t r = 0; r < rowSize; ++r)
                ret.data[r][c] = data[r][c];
        }
        for (size_t c = 0; c < other.colSize; ++c)
        {
            ret.colNames[c + colSize] = other.colNames[c];
            for (size_t r = 0; r < other.rowSize; ++r)
                ret.data[r][c + colSize] = other.data[r][c];
        }
        if (0 == size())
            for (size_t r = 0; r < other.rowSize; ++r)
                ret.rowNames[r] = other.rowNames[r];
        else
            for (size_t r = 0; r < rowSize; ++r)
                ret.rowNames[r] = rowNames[r];

        return ret;
    }
    default:
        cerr << "Error: Invalid Axis provided for concatenation operation. " << endl;
        throw invalid_argument("Invalid Axis for concatenation operation.");
    }
}
template <typename T> Mat<T> Mat<T>::transpose() const
{
    Mat<T> ret(colSize, rowSize);
    for (size_t c = 0; c < rowSize; ++c)
    {
        ret.colNames[c] = rowNames[c];
        for (size_t r = 0; r < colSize; ++r)
            ret.data[r][c] = data[c][r];
    }
    for (size_t r = 0; r < colSize; ++r)
        ret.rowNames[r] = colNames[r];

    return ret;
}
template <typename T> void Mat<T>::swap(const size_t a, const size_t b, const Axis axis)
{
    using namespace std;

    switch (axis)
    {
    case Axis::row: {
        if (a >= rowSize || b >= rowSize)
        {
            cerr << "Error: Row index out of bounds. Provided indices: a = " << a << ", b = " << b
                 << ", rowSize = " << rowSize << "." << endl;
            throw out_of_range("Row index out of bounds.");
        }

        std::swap(data[a], data[b]);
        std::swap(rowNames[a], rowNames[b]);

        this->refresh();
        break;
    }
    case Axis::col: {
        if (a >= colSize || b >= colSize)
        {
            cerr << "Error: Column index out of bounds. Provided indices: a = " << a << ", b = " << b
                 << ", colSize = " << colSize << "." << endl;
            throw out_of_range("Column index out of bounds.");
        }

        for (size_t r = 0; r < rowSize; ++r)
        {
            std::swap(data[r][a], data[r][b]);
            std::swap(colNames[a], colNames[b]);
        }

        this->refresh();
        break;
    }
    default:
        cerr << "Error: Invalid Axis provided for swap operation. " << endl;
        throw invalid_argument("Invalid Axis for swap operation.");
    }
}
template <typename T> void Mat<T>::drop(const size_t i, const Axis axis)
{
    using namespace std;

    switch (axis)
    {
    case Axis::row: {
        if (i >= rowSize)
        {
            cerr << "Error: Row index out of bounds. Provided index: " << i << ", rowSize = " << rowSize << "." << endl;
            throw out_of_range("Row index out of bounds.");
        }

        T     **newData     = new T *[rowSize - 1];
        string *newRowNames = new string[rowSize - 1];
        for (size_t r = 0, newRow = 0; r < rowSize; ++r)
            if (r != i)
            {
                newData[newRow]     = data[r];
                newRowNames[newRow] = rowNames[r];
                newRow++;
            }
            else
            {
                delete[] data[r];
            }
        delete[] data;
        delete[] rowNames;
        data     = newData;
        rowNames = newRowNames;
        rowSize -= 1;

        this->refresh();
        break;
    }
    case Axis::col: {
        if (i >= colSize)
        {
            cerr << "Error: Column index out of bounds. Provided index: " << i << ", colSize = " << colSize << "."
                 << endl;
            throw out_of_range("Column index out of bounds.");
        }

        T     **newData     = new T *[rowSize];
        string *newColNames = new string[colSize];
        for (size_t r = 0; r < rowSize; ++r)
        {
            newData[r] = new T[colSize - 1];
            for (size_t c = 0, newCol = 0; c < colSize; ++c)
                if (c != i)
                {
                    newData[r][newCol] = data[r][c];
                    newCol++;
                }
        }
        for (size_t c = 0, newCol = 0; c < colSize; ++c)
            if (c != i)
            {
                newColNames[newCol] = colNames[c];
                newCol++;
            }
        delete[] data;
        delete[] colNames;
        data     = newData;
        colNames = newColNames;
        colSize -= 1;

        this->refresh();
        break;
    }
    default:
        cerr << "Error: Invalid Axis provided for drop operation. " << endl;
        throw invalid_argument("Invalid Axis for drop operation.");
    }
}

// algorithm operation
template <typename T> void Mat<T>::sort(const size_t i, const Order order, const Axis axis)
{
    using namespace std;

    switch (axis)
    {
    case Axis::row: {
        if (i >= rowSize)
        {
            cerr << "Error: Row index out of bounds. Provided index: " << i << ", rowSize = " << rowSize << "." << endl;
            throw out_of_range("Row index out of bounds.");
        }

        vector<size_t> index(rowSize);
        for (size_t r = 0; r < rowSize; ++r)
            index[r] = r;
        std::sort(index.begin(), index.end(), [this, i, order](size_t a, size_t b) {
            return (order == Order::asce) ? (data[a][i] < data[b][i]) : (data[a][i] > data[b][i]);
        });

        T     **newData     = new T *[rowSize];
        string *newRowNames = new string[rowSize];
        for (size_t r = 0; r < rowSize; ++r)
        {
            newData[r]     = data[index[r]];
            newRowNames[r] = rowNames[index[r]];
        }
        delete[] data;
        delete[] rowNames;
        data     = newData;
        rowNames = newRowNames;

        this->refresh();
        break;
    }
    case Axis::col: {
        if (i >= colSize)
        {
            cerr << "Error: Column index out of bounds. Provided index: " << i << ", colSize = " << colSize << "."
                 << endl;
            throw out_of_range("Column index out of bounds.");
        }

        vector<size_t> index(colSize);
        for (size_t c = 0; c < colSize; ++c)
            index[c] = c;
        std::sort(index.begin(), index.end(), [this, i, order](size_t a, size_t b) {
            return (order == Order::asce) ? (data[i][a] < data[i][b]) : (data[i][a] > data[i][b]);
        });

        T     **newData     = new T *[rowSize];
        string *newColNames = new string[colSize];
        for (size_t r = 0; r < rowSize; ++r)
        {
            newData[r] = new T[colSize];
            for (size_t c = 0; c < colSize; ++c)
                newData[r][c] = data[r][index[c]];
            delete[] data[r];
        }
        for (size_t c = 0; c < colSize; ++c)
            newColNames[c] = colNames[index[c]];
        delete[] data;
        delete[] colNames;
        data     = newData;
        colNames = newColNames;

        this->refresh();
        break;
    }
    default:
        cerr << "Error: Invalid AxisType provided for sort operation." << endl;
        throw invalid_argument("Invalid AxisType for sort operation.");
    }
}
template <typename T> std::vector<std::pair<size_t, size_t>> Mat<T>::find(const T value) const
{
    using namespace std;

    vector<std::pair<size_t, size_t>> ret;
    for (size_t r = 0; r < rowSize; ++r)
        for (size_t c = 0; c < colSize; ++c)
            if (data[r][c] == value) ret.push_back(make_pair(r, c));
    return ret;
}
template <typename T> size_t Mat<T>::count(const T val) const
{
    size_t count = 0;
    for (size_t r = 0; r < rowSize; ++r)
        for (size_t c = 0; c < colSize; ++c)
            if (data[r][c] == val) count++;
    return count;
}
template <typename T> std::unordered_map<T, size_t> Mat<T>::count() const
{
    using namespace std;

    unordered_map<T, size_t> ret;

    for (size_t r = 0; r < rowSize; ++r)
        for (size_t c = 0; c < colSize; ++c)
        {
            ++ret[data[r][c]];
        }

    return ret;
}
template <typename T> Mat<T> Mat<T>::unique() const
{
    using namespace std;

    set<T> uniqueElements;
    for (size_t r = 0; r < rowSize; ++r)
        for (size_t c = 0; c < colSize; ++c)
            uniqueElements.insert(data[r][c]);

    Mat<T> ret(1, uniqueElements.size());
    size_t i = 0;
    for (const auto &e : uniqueElements)
    {
        ret.data[0][i++] = e;
    }

    return ret;
}

// matrix generation
template <typename T> Mat<T> Mat<T>::identity(const size_t size)
{
    Mat<T> ret(size, size);
    for (size_t i = 0; i < size; ++i)
        ret.data[i][i] = 1;

    return ret;
}

// tool functions
template <typename T> size_t Mat<T>::name2index(const std::string &name, const Axis axis) const
{
    using namespace std;

    if (name.empty())
    {
        cerr << "Error: name cannot be empty." << endl;
        throw invalid_argument("Name cannot be empty.");
    }

    switch (axis)
    {
    case Axis::row: {
        for (size_t r = 0; r < rowSize; ++r)
        {
            if (rowNames[r] == name)
            {
                return r;
            }
        }
        cerr << "Error: Row name not found: " << name << endl;
        throw invalid_argument("Row name not found.");
        break;
    }
    case Axis::col: {
        for (size_t c = 0; c < colSize; ++c)
        {
            if (colNames[c] == name)
            {
                return c;
            }
        }
        cerr << "Error: Column name not found: " << name << endl;
        throw invalid_argument("Column name not found.");
        break;
    }
    default: {
        cerr << "Error: Invalid AxisType provided for name2index operation." << endl;
        throw invalid_argument("Invalid AxisType for name2index operation.");
        break;
    }
    }
}

// display
template <typename T> void display(const Mat<T> &mat, const NameFlagType nameFlag, const size_t n_rows)
{
    using namespace std;

    const int width_val     = 15;
    const int width_rowName = 10;
    const int precision     = 5;

    cout << "Shape: " << mat.size(Axis::row) << " x " << mat.size(Axis::col) << endl;

    if (CheckFlag(nameFlag, NameFlag::col_name))
    {
        if (CheckFlag(nameFlag, NameFlag::row_name)) cout << fixed << setw(width_rowName) << " ";

        for (size_t c = 0; c < mat.size(Axis::col); ++c)
            cout << fixed << setw(width_val) << mat.iloc_name(c, Axis::col);
        cout << endl;
    }
    for (size_t r = 0; r < mat.size(Axis::row); ++r)
    {
        if (r >= n_rows) return; // limit output lines

        if (CheckFlag(nameFlag, row_name)) cout << fixed << setw(width_rowName) << mat.iloc_name(r, Axis::row);

        for (size_t c = 0; c < mat.size(Axis::col); ++c)
        {
            const T val = mat.iloc(r, c);
            if constexpr (is_same<T, string>::value)
                cout << setw(width_val) << val;
            else if constexpr (is_floating_point<T>::value)
            {
                if (val == static_cast<int>(val))
                    cout << fixed << setw(width_val) << static_cast<int>(val);
                else
                    cout << fixed << setprecision(precision) << setw(width_val) << val;
            }
            else
                cout << setw(width_val) << val;
        }
        cout << endl;
    }
}
template <typename T> void display_rainbow(const Mat<T> &mat, const NameFlagType nameFlag, const size_t n_rows)
{
    using namespace std;

    const int width_val     = 15;
    const int width_rowName = 10;
    const int precision     = 5;
    cout << "Shape: " << mat.size(Axis::row) << " x " << mat.size(Axis::col) << endl;
    if (CheckFlag(nameFlag, col_name))
    {
        if (CheckFlag(nameFlag, row_name)) cout << fixed << setw(width_rowName) << " ";

        for (size_t c = 0; c < mat.size(Axis::col); ++c)
        {
            cout << "\033[" << (c % 2 == 0 ? "32m" : "34m"); // alternate text color
            cout << fixed << setw(width_val) << mat.iloc_name(c, Axis::col);
        }

        cout << "\033[0m" << endl; // reset colors
    }
    for (size_t r = 0; r < mat.size(Axis::row); ++r)
    {
        if (r >= n_rows) return; // limit output lines

        cout << "\033[" << (r % 2 == 0 ? "48;5;235" : "48;5;240") << "m"; // alternate background color

        if (CheckFlag(nameFlag, row_name)) cout << fixed << setw(width_rowName) << mat.iloc_name(r, Axis::row);

        for (size_t c = 0; c < mat.size(Axis::col); ++c)
        {
            cout << "\033[" << (c % 2 == 0 ? "32m" : "34m"); // alternate text color
            const T val = mat.iloc(r, c);
            if constexpr (is_same<T, string>::value)
                cout << setw(width_val) << val;
            else if constexpr (is_floating_point<T>::value)
            {
                if (val == static_cast<int>(val))
                    cout << fixed << setw(width_val) << static_cast<int>(val);
                else
                    cout << fixed << setprecision(precision) << setw(width_val) << val;
            }
            else
                cout << setw(width_val) << val;
        }

        cout << "\033[0m" << endl; // reset colors
    }
}
} // namespace TL
#endif // MAT_HPP