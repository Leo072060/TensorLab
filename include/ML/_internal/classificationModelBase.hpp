#ifndef CLASSIFICATION_MODEL_BASE_HPP
#define CLASSIFICATION_MODEL_BASE_HPP

#include "_internal/managed.hpp"
#include "mat/mat.hpp"

namespace TL
{
namespace _internal
{
template <class T = double> class ClassificationModelBase : public ManagedClass
{
    // lifecycle management
  protected:
    ClassificationModelBase();
    ClassificationModelBase(const ClassificationModelBase<T> &other);
    ClassificationModelBase(ClassificationModelBase<T> &&other) noexcept;
    ClassificationModelBase<T> &operator=(const ClassificationModelBase<T> &rhs);
    ClassificationModelBase<T> &operator=(ClassificationModelBase<T> &&rhs) noexcept;
    ~ClassificationModelBase() {}

    // interface functions
  public:
    virtual void             train(const Mat<T> &x, const Mat<std::string> &y) = 0;
    virtual Mat<std::string> predict(const Mat<T> &x) const                    = 0;

    // for polymorphism
    virtual std::shared_ptr<ClassificationModelBase> clone() const = 0;
};

// lifecycle management
template <class T>
ClassificationModelBase<T>::ClassificationModelBase()
    : ManagedClass()
{
}
template <class T>
ClassificationModelBase<T>::ClassificationModelBase(const ClassificationModelBase<T> &other)
    : ManagedClass(other)
{
}
template <class T>
ClassificationModelBase<T>::ClassificationModelBase(ClassificationModelBase<T> &&other) noexcept
    : ClassificationModelBase(std::move(other))
{
}
template <class T>
ClassificationModelBase<T> &ClassificationModelBase<T>::operator=(const ClassificationModelBase<T> &rhs)
{
    ManagedClass::operator=(rhs);
    return *this;
}
template <class T>
ClassificationModelBase<T> &ClassificationModelBase<T>::operator=(ClassificationModelBase<T> &&rhs) noexcept
{
    using namespace std;
    ManagedClass::operator=(move(rhs));
    return *this;
}
} // namespace _internal
} // namespace TL
#endif // CLASSIFICATION_MODEL_BASE_HPP