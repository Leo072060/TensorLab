#ifndef BINARY_CLASSIFICATION_MODEL_BASE_HPP
#define BINARY_CLASSIFICATION_MODEL_BASE_HPP

#include <vector>

#include "ML/_internal/classificationModelBase.hpp"
#include "_internal/managed.hpp"
#include "mat/mat.hpp"

namespace TL
{
namespace _internal
{
enum BinaryToMultiMethod : int
{
    OneVsRest,
    OneVsOne
};

template <typename T> class BinaryClassificationModelBase : public ClassificationModelBase<T>
{
    // lifecycle management
  protected:
    BinaryClassificationModelBase();
    BinaryClassificationModelBase(const BinaryClassificationModelBase<T> &other);
    BinaryClassificationModelBase(BinaryClassificationModelBase<T> &&other) noexcept;
    BinaryClassificationModelBase<T> &operator=(const BinaryClassificationModelBase<T> &rhs);
    BinaryClassificationModelBase<T> &operator=(BinaryClassificationModelBase<T> &&rhs) noexcept;
    ~BinaryClassificationModelBase() {}

    // interface functions
  public:
    void                train(const Mat<T> &x, const Mat<std::string> &y) override;
    Mat<std::string>    predict(const Mat<T> &x) const override;
    std::vector<Mat<T>> get_thetas() const;

    // hook functions
  protected:
    virtual Mat<T>           train_binary(const Mat<T> &x, const Mat<std::string> &y)   = 0;
    virtual Mat<std::string> predict_binary(const Mat<T> &x, const Mat<T> &theta) const = 0;

    // internal implementation functions
  private:
    void        classify_labels(const Mat<T> &x, const Mat<std::string> &y);
    void        generate_ecoc(const BinaryToMultiMethod method);
    void        train_ecoc();
    std::string predict_ecoc(const Mat<T> &x) const;

  public:
    static size_t hammingDistance(const Mat<T> &lhs, const Mat<T> &rhs);

    // model parameters
  public:
    BinaryToMultiMethod binary2multi;

  protected:
    ManagedVal<Mat<std::string>> managed_labels;

  private:
    mutable ManagedVal<size_t>                        managed_dimension;
    mutable ManagedVal<BinaryToMultiMethod>           managed_binary2multi;
    mutable ManagedVal<std::vector<Mat<T>>>           managed_thetas;
    mutable ManagedVal<Mat<int>>                      managed_ecoc;
    mutable ManagedVal<std::vector<Mat<T>>>           managed_xs;
    mutable ManagedVal<std::vector<Mat<std::string>>> managed_ys;
};

// lifecycle management
template <typename T>
BinaryClassificationModelBase<T>::BinaryClassificationModelBase()
    : ClassificationModelBase<T>()
    , binary2multi(OneVsRest)
    , managed_labels(this->administrator)
    , managed_dimension(this->administrator)
    , managed_binary2multi(this->administrator)
    , managed_thetas(this->administrator)
    , managed_ecoc(this->administrator)
    , managed_xs(this->administrator)
    , managed_ys(this->administrator)
{
}
template <typename T>
BinaryClassificationModelBase<T>::BinaryClassificationModelBase(const BinaryClassificationModelBase<T> &other)
    : ClassificationModelBase<T>(other)
    , binary2multi(OneVsRest)
    , managed_labels(this->administrator, other.administrator, other.managed_labels)
    , managed_dimension(this->administrator, other.administrator, other.managed_dimension)
    , managed_binary2multi(this->administrator, other.administrator, other.managed_binary2multi)
    , managed_thetas(this->administrator, other.administrator, managed_thetas)
    , managed_ecoc(this->administrator, other.administrator, managed_ecoc)
    , managed_xs(this->administrator, other.administrator, other.managed_xs)
    , managed_ys(this->administrator, other.administrator, other.managed_ys)
{
}
template <typename T>
BinaryClassificationModelBase<T>::BinaryClassificationModelBase(BinaryClassificationModelBase<T> &&other) noexcept
    : ClassificationModelBase<T>(std::move(other))
    , binary2multi(OneVsRest)
    , managed_labels(this->administrator, other.administrator, other.managed_labels)
    , managed_dimension(this->administrator, other.administrator, other.managed_dimension)
    , managed_binary2multi(this->administrator, other.administrator, other.managed_binary2multi)
    , managed_thetas(this->administrator, other.administrator, managed_thetas)
    , managed_ecoc(this->administrator, other.administrator, managed_ecoc)
    , managed_xs(this->administrator, other.administrator, other.managed_xs)
    , managed_ys(this->administrator, other.administrator, other.managed_ys)
{
}
template <typename T>
BinaryClassificationModelBase<T> &BinaryClassificationModelBase<T>::operator=(
    const BinaryClassificationModelBase<T> &rhs)
{
    ClassificationModelBase<T>::operator=(rhs);
    managed_labels.copy(this->administrator, rhs.administrator, rhs.managed_labels);
    managed_dimension.copy(this->administrator, rhs.administrator, rhs.managed_dimension);
    managed_binary2multi.copy(this->administrator, rhs.administrator, rhs.managed_binary2multi);
    managed_thetas.copy(this->administrator, rhs.administrator, rhs.managed_thetas);
    managed_ecoc.copy(this->administrator, rhs.administrator, rhs.managed_ecoc);
    managed_xs.copy(this->administrator, rhs.administrator, rhs.managed_xs);
    managed_ys.copy(this->administrator, rhs.administrator, rhs.managed_ys);
    return *this;
}
template <typename T>
BinaryClassificationModelBase<T> &BinaryClassificationModelBase<T>::operator=(
    BinaryClassificationModelBase<T> &&rhs) noexcept
{
    using namespace std;
    ManagedClass::operator=(move(rhs));
    managed_labels.copy(this->administrator, rhs.administrator, rhs.managed_labels);
    managed_dimension.copy(this->administrator, rhs.administrator, rhs.managed_dimension);
    managed_binary2multi.copy(this->administrator, rhs.administrator, rhs.managed_binary2multi);
    managed_thetas.copy(this->administrator, rhs.administrator, rhs.managed_thetas);
    managed_ecoc.copy(this->administrator, rhs.administrator, rhs.managed_ecoc);
    managed_xs.copy(this->administrator, rhs.administrator, rhs.managed_xs);
    managed_ys.copy(this->administrator, rhs.administrator, rhs.managed_ys);
    return *this;
}

// interface functions
template <typename T> void BinaryClassificationModelBase<T>::train(const Mat<T> &x, const Mat<std::string> &y)
{
    using namespace std;

    if (y.size(Axis::col) != 1)
    {
        cerr << "Error: The label matrix must have exactly one column." << endl;
        throw invalid_argument("The label matrix must have exactly one column.");
    }
    if (x.size(Axis::row) != y.size(Axis::row))
    {
        cerr << "Error: The number of rows in the input matrix x (" << x.size(Axis::row)
             << ") does not match the number of rows in the label matrix y (" << y.size(Axis::row) << ")." << endl;
        throw invalid_argument(
            "The number of rows in the input matrix x must match the number of rows in the label matrix y.");
    }

    Mat<string> labels = y.unique();
    labels.sort(0, Order::asce, Axis::col);
    this->record(managed_labels, labels);
    classify_labels(x, y);

    switch (binary2multi)
    {
    case OneVsRest:
    case OneVsOne:
        generate_ecoc(binary2multi);
        train_ecoc();
        break;

    default:
        cerr << "Error: Invalid binary-to-multi classification method." << endl;
        throw invalid_argument("Invalid binary-to-multi classification method.");
    }

    this->record(managed_dimension, x.size(Axis::col));
    this->record(managed_binary2multi, binary2multi);
}
template <typename T> Mat<std::string> BinaryClassificationModelBase<T>::predict(const Mat<T> &x) const
{
    using namespace std;

    if (!managed_thetas.isReadable())
    {
        cerr << "Error: The model cannot predict before the model has been trained." << endl;
        throw runtime_error("The model has not been trained.");
    }
    if (managed_dimension.read() != x.size(Axis::col))
    {
        cerr << "Error: Input size mismatch. Expected input with " << managed_dimension.read() << " columns, but got "
             << x.size(Axis::col) << " columns." << endl;
        throw invalid_argument("Input size does not match the expected dimensions.");
    }

    Mat<string> ret(x.size(Axis::row), 1);
    for (size_t r = 0; r < x.size(Axis::row); ++r)
    {
        Mat<T> single_x = x.iloc(r, Axis::row);
        string pred;
        switch (managed_binary2multi.read())
        {
        case OneVsRest:
        case OneVsOne:
            pred = predict_ecoc(single_x);
            break;
        default:
            cerr << "Error: Invalid binary-to-multi classification method." << endl;
            throw invalid_argument("Invalid binary-to-multi classification method.");
        }
        ret.iloc(r, 0) = pred;
    }

    return ret;
}
template <class T> std::vector<Mat<T>> BinaryClassificationModelBase<T>::get_thetas() const
{
    using namespace std;

    if (!managed_thetas.isReadable())
    {
        cerr << "Error: The model parameters (thetas) cannot be read before the model has been trained." << endl;
        throw runtime_error("The model has not been trained.");
    }
    return managed_thetas.read();
}

// internal implementation functions
template <typename T> void BinaryClassificationModelBase<T>::classify_labels(const Mat<T> &x, const Mat<std::string> &y)
{
    using namespace std;

    vector<Mat<T>>      xs;
    vector<Mat<string>> ys;
    for (size_t c = 0; c < managed_labels.read().size(Axis::col); ++c)
    {
        vector<pair<size_t, size_t>> loc = y.find(managed_labels.read().iloc(0, c));
        vector<size_t>               index;
        for (const auto &e : loc)
            index.push_back(e.first);
        xs.push_back(x.extract(index, Axis::row));
        ys.push_back(y.extract(index, Axis::row));
    }
    this->record(managed_xs, xs);
    this->record(managed_ys, ys);
}
template <typename T> void BinaryClassificationModelBase<T>::generate_ecoc(const BinaryToMultiMethod method)
{
    using namespace std;

    switch (method)
    {
    case OneVsRest: {
        size_t   numClasses = managed_labels.read().size(Axis::col);
        Mat<int> ecocMatrix(numClasses, numClasses);
        ecocMatrix = -1;

        for (size_t i = 0; i < numClasses; ++i)
            ecocMatrix.iloc(i, i) = 1;

        this->record(managed_ecoc, ecocMatrix);
        break;
    }
    case OneVsOne: {
        size_t   numClasses = managed_labels.read().size(Axis::col);
        size_t   pairCount  = (numClasses * (numClasses - 1)) / 2;
        Mat<int> ecocMatrix(numClasses, pairCount);

        size_t c = 0;
        for (size_t i = 0; i < numClasses; ++i)
        {
            for (size_t j = i + 1; j < numClasses; ++j)
            {
                ecocMatrix.iloc(i, c) = 0;
                ecocMatrix.iloc(j, c) = 1;
                ++c;
            }
        }

        this->record(managed_ecoc, ecocMatrix);
        break;
    }
    default:
        cerr << "Error: Invalid BinaryToMultiMethod provided." << endl;
        throw invalid_argument("Invalid BinaryToMultiMethod.");
    }
}
template <typename T> void BinaryClassificationModelBase<T>::train_ecoc()
{
    using namespace std;

    vector<Mat<T>> thetas;
    const Mat<int> ecoc = managed_ecoc.read();
    for (size_t c = 0; c < ecoc.size(Axis::col); ++c)
    {
        Mat<T>      x;
        Mat<string> y;
        for (size_t r = 0; r < ecoc.size(Axis::row); ++r)
        {
            if (ecoc.iloc(r, c) != 0)
            {
                x                 = x.concat(managed_xs.read()[r], Axis::row);
                Mat<string> tmp_y = managed_ys.read()[r];
                if (1 == ecoc.iloc(r, c))
                    tmp_y = "T";
                else if (-1 == ecoc.iloc(r, c))
                    tmp_y = "F";
                y = y.concat(tmp_y, Axis::row);
            }
        }
        thetas.push_back(train_binary(x, y));
    }
    this->record(managed_thetas, thetas);
}
template <typename T> std::string BinaryClassificationModelBase<T>::predict_ecoc(const Mat<T> &x) const
{
    Mat<int> ecoc = managed_ecoc.read();
    Mat<int> pred_ecoc(1, ecoc.size(Axis::col));
    size_t   c = 0;
    for (const auto &e : managed_thetas.read())
        pred_ecoc.iloc(0, c++) = (predict_binary(x, e).iloc(0, 0) == "T") ? 1 : -1;

    size_t minHammingDistance = BinaryClassificationModelBase<int>::hammingDistance(ecoc.iloc(0, Axis::row), pred_ecoc);
    size_t pred_r             = 0;
    for (size_t r = 1; r < managed_ecoc.read().size(Axis::row); ++r)
    {
        size_t distance = BinaryClassificationModelBase<int>::hammingDistance(ecoc.iloc(r, Axis::row), pred_ecoc);
        if (distance < minHammingDistance)
        {
            minHammingDistance = distance;
            pred_r             = r;
        }
    }

    return managed_labels.read().iloc(0, pred_r);
}
template <typename T> size_t BinaryClassificationModelBase<T>::hammingDistance(const Mat<T> &lhs, const Mat<T> &rhs)
{
    using namespace std;

    if (lhs.size(Axis::row) != rhs.size(Axis::row) || lhs.size(Axis::col) != rhs.size(Axis::col))
    {
        cerr << "Error: Matrix dimensions do not match for Hamming distance calculation. "
             << "lhs matrix is (" << lhs.size(Axis::row) << "x" << lhs.size(Axis::col) << "), while rhs matrix is ("
             << rhs.size(Axis::row) << "x" << rhs.size(Axis::col) << ")." << endl;
        throw invalid_argument("Matrices must have the same dimensions for Hamming distance.");
    }

    size_t ret = 0;
    for (size_t r = 0; r < lhs.size(Axis::row); ++r)
        for (size_t c = 0; c < lhs.size(Axis::col); ++c)
            if (lhs.iloc(r, c) != rhs.iloc(r, c)) ++ret;

    return ret;
}
} // namespace _internal
} // namespace TL
#endif // BINARY_CLASSIFICATION_MODEL_BASE_HPP
