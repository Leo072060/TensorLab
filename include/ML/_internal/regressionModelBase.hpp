#ifndef REGRESSION_MODELBASE_HPP
#define REGRESSION_MODELBASE_HPP

#include <vector>

#include "_internal/managed.hpp"
#include "mat/mat.hpp"

namespace TL
{
namespace _internal
{
template <class T = double> class RegressionModelBase : public ManagedClass
{
    // lifecycle management
  protected:
    RegressionModelBase();
    RegressionModelBase(const RegressionModelBase<T> &other);
    RegressionModelBase(RegressionModelBase<T> &&other) noexcept;
    RegressionModelBase<T> &operator=(const RegressionModelBase<T> &rhs);
    RegressionModelBase<T> &operator=(RegressionModelBase<T> &&rhs) noexcept;
    ~RegressionModelBase() = default;

    // interface functions
  public:
    void   train(const Mat<T> &x, const Mat<T> &y);
    Mat<T> predict(const Mat<T> &x) const;
    Mat<T> get_thetas() const;

    // hook functions
  protected:
    virtual Mat<T> train_(const Mat<T> &x, const Mat<T> &y)              = 0;
    virtual Mat<T> predict_(const Mat<T> &x, const Mat<T> &thetas) const = 0;

    // for polymorphism
  public:
    virtual std::shared_ptr<RegressionModelBase<T>> clone() const = 0;

    // model parameters
  private:
    mutable ManagedVal<Mat<T>> managed_thetas;
};

// lifecycle management
template <class T>
RegressionModelBase<T>::RegressionModelBase()
    : ManagedClass()
    , managed_thetas(this->administrator)
{
}
template <class T>
RegressionModelBase<T>::RegressionModelBase(const RegressionModelBase<T> &other)
    : ManagedClass(other)
    , managed_thetas(this->administrator, other.administrator, other.managed_thetas)
{
}
template <class T>
RegressionModelBase<T>::RegressionModelBase(RegressionModelBase<T> &&other) noexcept
    : ManagedClass(std::move(other))
    , managed_thetas(this->administrator, other.administrator, other.managed_thetas)
{
}
template <class T> RegressionModelBase<T> &RegressionModelBase<T>::operator=(const RegressionModelBase<T> &rhs)
{
    ManagedClass::operator=(rhs);
    managed_thetas.copy(this->administrator, rhs.administrator, rhs.managed_thetas);
    return *this;
}
template <class T> RegressionModelBase<T> &RegressionModelBase<T>::operator=(RegressionModelBase<T> &&rhs) noexcept
{
    using namespace std;
    ManagedClass::operator=(move(rhs));
    managed_thetas.copy(this->administrator, rhs.administrator, rhs.managed_thetas);
    return *this;
}

// interface functions
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
} // namespace _internal
} // namespace TL
#endif // REGRESSION_MODELBASE_HPP
