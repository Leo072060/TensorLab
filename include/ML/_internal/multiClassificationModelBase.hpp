#ifndef MULTI_CLASSIFICATION_MODEL_BASE_HPP
#define MULTI_CLASSIFICATION_MODEL_BASE_HPP

#include "ML/_internal/classificationModelBase.hpp"
#include "_internal/managed.hpp"
#include "mat/mat.hpp"

namespace TL
{
namespace _internal
{
template <class T> class MultiClassificationModelBase : public ClassificationModelBase<T>
{
    // lifecycle management
  protected:
    MultiClassificationModelBase();
    MultiClassificationModelBase(const MultiClassificationModelBase<T> &other);
    MultiClassificationModelBase(MultiClassificationModelBase<T> &&other) noexcept;
    MultiClassificationModelBase<T> &operator=(const MultiClassificationModelBase<T> &rhs);
    MultiClassificationModelBase<T> &operator=(MultiClassificationModelBase<T> &&rhs) noexcept;
    ~MultiClassificationModelBase();

    // interface functions
  public:
    void             train(const Mat<T> &x, const Mat<std::string> &y) override;
    Mat<std::string> predict(const Mat<T> &x) const override;
    Mat<double>           get_thetas() const;

    // hook functions
  protected:
    virtual Mat<double>           train_multi(const Mat<T> &x, const Mat<std::string> &y)   = 0;
    virtual Mat<std::string> predict_multi(const Mat<T> &x, const Mat<double> &theta) const = 0;

    // model parammeters
  protected:
    mutable ManagedVal<Mat<std::string>> managed_labels;

  private:
    mutable ManagedVal<Mat<double>> managed_thetas;
};

// lifecycle management
template <class T>
MultiClassificationModelBase<T>::MultiClassificationModelBase()
    : ClassificationModelBase<T>()
    , managed_thetas(this->administrator)
    , managed_labels(this->administrator)
{
}
template <class T>
MultiClassificationModelBase<T>::MultiClassificationModelBase(const MultiClassificationModelBase<T> &other)
    : ClassificationModelBase<T>(other)
    , managed_thetas(this->administrator, other.administrator, other.managed_thetas)
    , managed_labels(this->administrator, other.administrator, other.managed_labels)
{
}
template <class T>
MultiClassificationModelBase<T>::MultiClassificationModelBase(MultiClassificationModelBase<T> &&other) noexcept
    : ClassificationModelBase<T>(std::move(other))
    , managed_thetas(this->administrator, other.administrator, other.managed_thetas)
    , managed_labels(this->administrator, other.administrator, other.managed_labels)
{
}
template <class T>
MultiClassificationModelBase<T> &MultiClassificationModelBase<T>::operator=(const MultiClassificationModelBase<T> &rhs)
{
    ManagedClass::operator=(rhs);
    managed_thetas.copy(this->administrator, rhs.administrator, rhs.managed_thetas);
    managed_labels.copy(this->administrator, rhs.administrator, rhs.managed_labels);
}
template <class T>
MultiClassificationModelBase<T> &MultiClassificationModelBase<T>::operator=(
    MultiClassificationModelBase<T> &&rhs) noexcept
{
    using namespace std;
    ManagedClass::operator=(move(rhs));
    managed_thetas.copy(this->administrator, rhs.administrator, rhs.managed_thetas);
    managed_labels.copy(this->administrator, rhs.administrator, rhs.managed_labels);
}
template <class T> MultiClassificationModelBase<T>::~MultiClassificationModelBase() {}

// interface functions
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
template <class T> Mat<double> MultiClassificationModelBase<T>::get_thetas() const
{
    using namespace std;

    if (!managed_thetas.isReadable())
    {
        cerr << "Error: The model parameters (thetas) cannot be read before the model has been trained." << endl;
        throw runtime_error("The model has not been trained.");
    }
    return managed_thetas.read();
}
} // namespace _internal
} // namespace TL
#endif // MULTI_CLASSIFICATION_MODEL_BASE_HPP