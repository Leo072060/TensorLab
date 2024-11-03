#ifndef MANAGED_HPP
#define MANAGED_HPP

#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "_internal/uncopyable.h"
#include "_internal/utility.hpp"

namespace TL
{
namespace _internal
{
#pragma region Administrator
using ADMINISTRATOR_ID = unsigned long;
class ManagedItem;
class Administrator : private Uncopyable
{
    friend class ManagedItem;

  public:
    Administrator()
        : ID(GlobalID++)
    {
    }
    ~Administrator() = default;

    void                         registerManagedItem(std::shared_ptr<ManagedItem> ptr_ManagedItem);
    size_t                       managedItemsSize() const;
    std::shared_ptr<ManagedItem> getManagedItem(const size_t i) const;

  private:
    const ADMINISTRATOR_ID                    ID;
    std::vector<std::shared_ptr<ManagedItem>> managedItemList;
    static ADMINISTRATOR_ID                   GlobalID; // a counter to assigned number
};
#pragma endregion

#pragma region ManagedItem
using PermissionType = uint8_t;
enum Permission : uint8_t
{
    lowest   = 0b11111111,
    readable = 0b11111110,
    writable = 0b11111101
};
class ManagedItem : private Uncopyable
{
  protected:
    explicit ManagedItem(Administrator &admin, const PermissionType perm = Permission::lowest);

  public:
    virtual ~ManagedItem() = default;

  public:
    bool           authenticate(const Administrator &admin) const;
    bool           checkPermission(const PermissionType perm) const;
    PermissionType getPermission() const;
    bool           isReadable() const;
    bool           isWritable() const;
    void           setPermission(const Administrator &admin, const PermissionType perm);
    void           addPermission(const Administrator &admin, const PermissionType perm);
    void           delPermission(const Administrator &admin, const PermissionType perm);
    virtual void   copy(const Administrator &admin, const Administrator &admin_other, const ManagedItem &other) = 0;

  protected:
    const ADMINISTRATOR_ID administrator_ID;
    PermissionType         permission;
};
#pragma endregion

#pragma region ManagedVal
template <class T> class ManagedVal;
template <class T> class ManagedValImpl : public ManagedItem
{
    friend class ManagedVal<T>;

  private:
    explicit ManagedValImpl(Administrator &admin, const PermissionType perm = Permission::lowest, const T &val = T());
    static std::shared_ptr<ManagedValImpl<T>> getInstance(Administrator       &admin,
                                                          const PermissionType perm = Permission::lowest,
                                                          const T             &val  = T());

  public:
    ~ManagedValImpl() override = default;

  public:
    T    read() const;
    T    read(const Administrator &admin) const;
    void write(const T &val);
    void copy(const Administrator &admin, const Administrator &admin_other, const ManagedItem &other) override;

  private:
    T value;
};
template <typename T>
ManagedValImpl<T>::ManagedValImpl(Administrator &admin, const PermissionType perm, const T &val)
    : ManagedItem(admin, perm)
    , value(val)
{
}
template <typename T> T ManagedValImpl<T>::read() const
{
    using namespace std;

    if (!this->checkPermission(readable))
    {
        cerr << "Insufficient permissions for read operation.";
        throw runtime_error("Permission denied.");
    }

    return value;
}
template <typename T> T ManagedValImpl<T>::read(const Administrator &admin) const
{
    using namespace std;
    if (!this->authenticate(admin))
    {
        cerr << "Error: Permission denied: administrator ID mismatch. " << endl;
        throw runtime_error("Permission denied.");
    }
    return value;
}
template <typename T> void ManagedValImpl<T>::write(const T &val)
{
    using namespace std;

    if (!this->checkPermission(writable))
    {
        cerr << "Insufficient permissions for write operation.";
        throw runtime_error("Permission denied.");
    }

    value = val;
}
template <typename T>
void ManagedValImpl<T>::copy(const Administrator &admin, const Administrator &admin_other, const ManagedItem &other)
{
    using namespace std;
    ManagedItem    &other_non_const = const_cast<ManagedItem &>(other);
    ManagedValImpl &other_cast      = static_cast<ManagedValImpl &>(other_non_const);
    PermissionType  perm            = other_cast.getPermission();
    other_cast.setPermission(admin_other, readable);
    this->setPermission(admin, writable);
    this->write(other_cast.read());
    other_cast.setPermission(admin_other, perm);
    this->setPermission(admin, perm);
}
template <typename T>
std::shared_ptr<ManagedValImpl<T>> ManagedValImpl<T>::getInstance(Administrator &admin, const PermissionType perm,
                                                                  const T &val)
{
    using namespace std;
    shared_ptr<ManagedValImpl<T>> new_managedValImpl(new ManagedValImpl<T>(admin, perm, val));
    admin.registerManagedItem(new_managedValImpl);
    return new_managedValImpl;
}

class ManagedClass;
template <class T> class ManagedVal : Uncopyable
{
    friend class ManagedClass;

  public:
    explicit ManagedVal(Administrator &admin);
    ManagedVal(Administrator &admin, const Administrator &admin_other, const ManagedVal<T> &managedVal_other);
    ~ManagedVal() = default;

    bool isReadable() const;
    T    read() const;
    void copy(const Administrator &admin, const Administrator &admin_other, const ManagedVal<T> &other);

  private:
    const std::shared_ptr<ManagedValImpl<T>> &operator->() const;

  private:
    const std::shared_ptr<ManagedValImpl<T>> ptr_managedValImpl;
};
template <typename T>
ManagedVal<T>::ManagedVal(Administrator &admin)
    : ptr_managedValImpl(ManagedValImpl<T>::getInstance(admin))
{
}
template <typename T>
ManagedVal<T>::ManagedVal(Administrator &admin, const Administrator &admin_other, const ManagedVal<T> &managedVal_other)
    : ptr_managedValImpl(
          ManagedValImpl<T>::getInstance(admin, managedVal_other->getPermission(), managedVal_other->read(admin_other)))
{
}
template <typename T> bool ManagedVal<T>::isReadable() const
{
    return ptr_managedValImpl->isReadable();
}
template <typename T> T ManagedVal<T>::read() const
{
    return ptr_managedValImpl->read();
}
template <typename T>
void ManagedVal<T>::copy(const Administrator &admin, const Administrator &admin_other, const ManagedVal<T> &other)
{
    ptr_managedValImpl->copy(admin, admin_other, *other.ptr_managedValImpl);
}
template <typename T> const std::shared_ptr<ManagedValImpl<T>> &ManagedVal<T>::operator->() const
{
    return ptr_managedValImpl;
}
#pragma endregion

#pragma region ManagedClass
class ManagedClass
{
  public:
    ManagedClass() = default;
    ManagedClass(const ManagedClass &other);
    ManagedClass(ManagedClass &&other) noexcept;
    ManagedClass &operator=(const ManagedClass &rhs);
    ManagedClass &operator=(ManagedClass &&rhs) noexcept;
    ~ManagedClass() = default;

  protected:
    template <typename T> void record(ManagedVal<T> &managedVal, const T &val) const;
    void                       refresh() const;
    void                       copyManagedVals(const ManagedClass &other) const;

  protected:
    Administrator administrator;
    mutable bool  isrefreshed = true;
};
template <typename T> void ManagedClass::record(ManagedVal<T> &managedVal, const T &val) const
{
    managedVal->setPermission(administrator, writable);
    managedVal->write(val);
    managedVal->setPermission(administrator, readable);
    isrefreshed = false;
}
#pragma endregion
} // namespace _internal
} // namespace TL
#endif // MANAGED_HPP
