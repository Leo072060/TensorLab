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
    Mat<T> get_theta() const;
    void   set_theta(const Mat<double>& theta);

    // hook functions
  protected:
    virtual Mat<double> train_(const Mat<T> &x, const Mat<T> &y)                   = 0;
    virtual Mat<T>      predict_(const Mat<T> &x, const Mat<double> &thetas) const = 0;

    // for polymorphism
  public:
    virtual std::shared_ptr<RegressionModelBase<T>> clone() const = 0;

    // model parameters
  private:
    mutable ManagedVal<Mat<double>> managed_theta;
};

// lifecycle management
template <class T>
RegressionModelBase<T>::RegressionModelBase()
    : ManagedClass()
    , managed_theta(this->administrator)
{
}
template <class T>
RegressionModelBase<T>::RegressionModelBase(const RegressionModelBase<T> &other)
    : ManagedClass(other)
    , managed_theta(this->administrator, other.administrator, other.managed_theta)
{
}
template <class T>
RegressionModelBase<T>::RegressionModelBase(RegressionModelBase<T> &&other) noexcept
    : ManagedClass(std::move(other))
    , managed_theta(this->administrator, other.administrator, other.managed_theta)
{
}
template <class T> RegressionModelBase<T> &RegressionModelBase<T>::operator=(const RegressionModelBase<T> &rhs)
{
    ManagedClass::operator=(rhs);
    managed_theta.copy(this->administrator, rhs.administrator, rhs.managed_theta);
    return *this;
}
template <class T> RegressionModelBase<T> &RegressionModelBase<T>::operator=(RegressionModelBase<T> &&rhs) noexcept
{
    using namespace std;
    ManagedClass::operator=(move(rhs));
    managed_theta.copy(this->administrator, rhs.administrator, rhs.managed_theta);
    return *this;
}

// interface functions
template <class T> void RegressionModelBase<T>::train(const Mat<T> &x, const Mat<T> &y)
{
    this->record(managed_theta, this->train_(x, y));
}
template <class T> Mat<T> RegressionModelBase<T>::predict(const Mat<T> &x) const
{
    using namespace std;

    if (!managed_theta.isReadable())
    {
        cerr << "Error: The model cannot predict before the model has been trained." << endl;
        throw runtime_error("The model has not been trained.");
    }

    return predict_(x, managed_theta.read());
}
template <class T> Mat<T> RegressionModelBase<T>::get_theta() const
{
    using namespace std;

    if (!managed_theta.isReadable())
    {
        cerr << "Error: The model parameters (thetas) cannot be read before the model has been trained." << endl;
        throw runtime_error("The model has not been trained.");
    }
    return managed_theta.read();
}
template <class T> void RegressionModelBase<T>::set_theta(const Mat<double>& theta) {
    record(managed_theta,theta);
}
} // namespace _internal
} // namespace TL
#endif // REGRESSION_MODELBASE_HPP
