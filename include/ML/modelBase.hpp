#ifndef MODELBASE_HPP
#define MODELBASE_HPP

#include <vector>

#include "kits/managed.hpp"
#include "mat/mat.hpp"

#pragma region RegressionModelBase
template <class T = double> class RegressionModelBase : public ManagedClass
{
  protected:
    RegressionModelBase();
    RegressionModelBase(const RegressionModelBase<T> &other);
    RegressionModelBase(RegressionModelBase<T> &&other) noexcept;
    RegressionModelBase<T> &operator=(const RegressionModelBase<T> &rhs);
    RegressionModelBase<T> &operator=(RegressionModelBase<T> &&rhs) noexcept;
    ~RegressionModelBase() = default;

  public:
    void   train(const Mat<T> &x, const Mat<T> &y);
    Mat<T> predict(const Mat<T> &x) const;
    Mat<T> get_thetas() const;

    ///////////////////// API ////////////////////////////
  public:
    virtual std::shared_ptr<RegressionModelBase<T>> clone() const = 0;

  protected:
    virtual Mat<T> train_(const Mat<T> &x, const Mat<T> &y)              = 0;
    virtual Mat<T> predict_(const Mat<T> &x, const Mat<T> &thetas) const = 0;
    /////////////////////////////////////////////////

  private:
    ManagedVal<Mat<T>> managed_thetas;
};
template <class T> RegressionModelBase<T>::RegressionModelBase() : ManagedClass(), managed_thetas(this->administrator)
{
}
template <class T>
RegressionModelBase<T>::RegressionModelBase(const RegressionModelBase<T> &other) : RegressionModelBase()
{
    this->copyManagedClass(other);
    this->copyManagedVal(managed_thetas, other.managed_thetas, other);
}
template <class T>
RegressionModelBase<T>::RegressionModelBase(RegressionModelBase<T> &&other) noexcept : RegressionModelBase()
{
    this->copyManagedClass(other);
    this->copyManagedVal(managed_thetas, other.managed_thetas, other);
}
template <class T> RegressionModelBase<T> &RegressionModelBase<T>::operator=(const RegressionModelBase<T> &rhs)
{
    this->copyManagedClass(rhs);
    this->copyManagedVal(managed_thetas, rhs.managed_thetas, rhs);
    return *this;
}
template <class T> RegressionModelBase<T> &RegressionModelBase<T>::operator=(RegressionModelBase<T> &&rhs) noexcept
{
    this->copyManagedClass(rhs);
    this->copyManagedVal(managed_thetas, rhs.managed_thetas, rhs);
    return *this;
}
template <class T> void RegressionModelBase<T>::train(const Mat<T> &x, const Mat<T> &y)
{
    this->record(managed_thetas, this->train_(x, y));
}
template <class T> Mat<T> RegressionModelBase<T>::predict(const Mat<T> &x) const
{
    using namespace std;

    if (!managed_thetas.isReadable())
    {
        cerr << "Error: The model cannot predict before the model has been trained." << endl;
        throw runtime_error("The model has not been trained.");
    }

    return predict_(x, managed_thetas.read());
}
template <class T> Mat<T> RegressionModelBase<T>::get_thetas() const
{
    using namespace std;

    if (!managed_thetas.isReadable())
    {
        cerr << "Error: The model parameters (thetas) cannot be read before the model has been trained." << endl;
        throw runtime_error("The model has not been trained.");
    }
    return managed_thetas.read();
}
#pragma endregion

#pragma region ClassificationModelBase
template <class T = double> class ClassificationModelBase : public ManagedClass
{
  protected:
    ClassificationModelBase() : ManagedClass() {}
    ClassificationModelBase(const ClassificationModelBase<T> &other) : ClassificationModelBase() {}
    ClassificationModelBase(ClassificationModelBase<T> &&other) noexcept : ClassificationModelBase() {}
    ~ClassificationModelBase() {}

  public:
    virtual void             train(const Mat<T> &x, const Mat<std::string> &y) = 0;
    virtual Mat<std::string> predict(const Mat<T> &x) const                    = 0;

    ///////////////////// API ////////////////////////////
  public:
    virtual std::shared_ptr<ClassificationModelBase> clone() const = 0;
    /////////////////////////////////////////////////
};
#pragma endregion

#pragma region MultiClassificationModelBase
template <class T> class MultiClassificationModelBase : public ClassificationModelBase<T>
{
  protected:
    MultiClassificationModelBase();
    MultiClassificationModelBase(const MultiClassificationModelBase<T> &other);
    MultiClassificationModelBase(MultiClassificationModelBase<T> &&other) noexcept;
    ~MultiClassificationModelBase();

  public:
    void             train(const Mat<T> &x, const Mat<std::string> &y) override;
    Mat<std::string> predict(const Mat<T> &x) const override;
    Mat<T>           get_thetas() const;

    ///////////////////// API ////////////////////////////
  protected:
    virtual Mat<T>           train_multi(const Mat<T> &x, const Mat<std::string> &y)   = 0;
    virtual Mat<std::string> predict_multi(const Mat<T> &x, const Mat<T> &theta) const = 0;
    /////////////////////////////////////////////////

  protected:
    ManagedVal<Mat<std::string>> managed_labels;

  private:
    ManagedVal<Mat<T>> managed_thetas;
};
template <class T>
MultiClassificationModelBase<T>::MultiClassificationModelBase()
    : ClassificationModelBase<T>(), managed_thetas(this->administrator), managed_labels(this->administrator)
{
}
template <class T>
MultiClassificationModelBase<T>::MultiClassificationModelBase(const MultiClassificationModelBase<T> &other)
    : MultiClassificationModelBase()
{
    this->copyIfReadable(managed_labels, other.managed_labels);
    this->copyIfReadable(managed_thetas, other.managed_thetas);
}
template <class T>
MultiClassificationModelBase<T>::MultiClassificationModelBase(MultiClassificationModelBase<T> &&other) noexcept
    : MultiClassificationModelBase()
{
    this->copyIfReadable(managed_labels, other.managed_labels);
    this->copyIfReadable(managed_thetas, other.managed_thetas);
}
template <class T> MultiClassificationModelBase<T>::~MultiClassificationModelBase() {}
template <class T> void MultiClassificationModelBase<T>::train(const Mat<T> &x, const Mat<std::string> &y)
{
    this->record(managed_thetas, this->train_multi(x, y));
}
template <class T> Mat<std::string> MultiClassificationModelBase<T>::predict(const Mat<T> &x) const
{
    using namespace std;

    if (!managed_thetas.isReadable())
    {
        cerr << "Error: The model cannot predict before the model has been trained." << endl;
        throw runtime_error("The model has not been trained.");
    }

    return predict_multi(x, managed_thetas.read());
}
template <class T> Mat<T> MultiClassificationModelBase<T>::get_thetas() const
{
    using namespace std;

    if (!managed_thetas.isReadable())
    {
        cerr << "Error: The model parameters (thetas) cannot be read before the model has been trained." << endl;
        throw runtime_error("The model has not been trained.");
    }
    return managed_thetas.read();
}
#pragma endregion

#pragma region BinaryClassificationModelBase
enum BinaryToMultiMethod : int
{
    OneVsRest,
    OneVsOne
};
template <typename T> class BinaryClassificationModelBase : public ClassificationModelBase<T>
{
  protected:
    BinaryClassificationModelBase()
        : ClassificationModelBase<T>(), managed_labels(this->administrator), managed_binary2multi(this->administrator),
          managed_thetas(this->administrator), managed_ecoc(this->administrator)
    {
    }
    BinaryClassificationModelBase(const BinaryClassificationModelBase<T> &other) : BinaryClassificationModelBase() {}
    BinaryClassificationModelBase(BinaryClassificationModelBase<T> &&other) noexcept : BinaryClassificationModelBase()
    {
    }
    ~BinaryClassificationModelBase() {}

  public:
    void                train(const Mat<T> &x, const Mat<std::string> &y) override;
    Mat<std::string>    predict(const Mat<T> &x) const override;
    std::vector<Mat<T>> get_thetas() const;

    ///////////////////// API ////////////////////////////
  protected:
    virtual Mat<T>           train_binary(const Mat<T> &x, const Mat<std::string> &y)   = 0;
    virtual Mat<std::string> predict_binary(const Mat<T> &x, const Mat<T> &theta) const = 0;
    /////////////////////////////////////////////////

  private:
    void        classify_labels(const Mat<T> &x, const Mat<std::string> &y);
    void        generate_ecoc(const BinaryToMultiMethod method);
    void        train_ecoc();
    std::string predict_ecoc(const Mat<T> &x) const;
    size_t      hammingDistance(const Mat<T> &lhs, const Mat<T> &rhs) const;

  public:
    BinaryToMultiMethod binary2multi;

  protected:
    ManagedVal<Mat<std::string>> managed_labels;

  private:
    ManagedVal<size_t>                        managed_dimension;
    ManagedVal<BinaryToMultiMethod>           managed_binary2multi;
    ManagedVal<std::vector<Mat<T>>>           managed_thetas;
    ManagedVal<Mat<int>>                      managed_ecoc;
    ManagedVal<std::vector<Mat<T>>>           managed_xs;
    ManagedVal<std::vector<Mat<std::string>>> managed_ys;
};
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
        ret.iloc(0, r) = pred;
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
template <typename T> void BinaryClassificationModelBase<T>::classify_labels(const Mat<T> &x, const Mat<std::string> &y)
{
    using namespace std;

    vector<Mat<T>>      xs;
    vector<Mat<string>> ys;
    for (size_t c = 0; c < managed_labels.read().size(Axis::row); ++c)
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
    const Mat<T>   ecoc = managed_ecoc.read();
    for (size_t c = 0; c < ecoc.size(Axis::col); ++c)
    {
        Mat<T>      x;
        Mat<string> y;
        for (size_t r = 0; r < ecoc.size(Axis::row); ++r)
        {
            if (ecoc.iloc(r, c) != 0)
            {
                x                 = x.concat(managed_xs.read()[r]);
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
        pred_ecoc.iloc(0, c++) = (ipredict_binary(x, e).iloc(0, 0) == "T") ? 1 : -1;

    size_t minHammingDistance = hammingDistance(ecoc.iloc(0, Axis::row), pred_ecoc);
    size_t pred_r             = 0;
    for (size_t r = 1; r < managed_ecoc.read().size(Axis::row); ++r)
    {
        size_t distance = hammingDistance(ecoc.iloc(0, Axis::row), pred_ecoc);
        if (distance < minHammingDistance)
        {
            minHammingDistance = distance;
            pred_r             = r;
        }
    }

    return managed_labels.read().iloc(0, pred_r);
}
template <typename T>
size_t BinaryClassificationModelBase<T>::hammingDistance(const Mat<T> &lhs, const Mat<T> &rhs) const
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
#pragma endregion

#endif // MODELBASE_HPP
