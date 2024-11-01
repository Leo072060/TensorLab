#ifndef MANAGED_HPP
#define MANAGED_HPP

#include <memory>
#include <stdexcept>
#include <vector>

using ADMINISTRATOR_ID = unsigned long;

using PermissionType = uint8_t;
enum Permission : uint8_t
{
    lowest   = 0b11111111,
    readable = 0b11111110,
    writable = 0b11111101
};

#define CHECK_FLAG(VAL, FLAG) ((VAL | FLAG) == FLAG)

class ManagedItem;
class Administrator
{
  public:
    Administrator() : ID(GlobalID++) {}

    void               registerConstManagedItem(const ManagedItem &managedItem) const;
    void               registerManagedItem(ManagedItem &managedItem) const;
    size_t             constManagedItemsSize() const;
    size_t             managedItemsSize() const;
    const ManagedItem &getConstManagedItem(const size_t i) const;
    ManagedItem       &getManagedItem(const size_t i) const;

  public:
    const ADMINISTRATOR_ID ID;

  private:
    mutable std::vector<std::unique_ptr<const ManagedItem>> constManagedItemList;
    mutable std::vector<std::unique_ptr<ManagedItem>>       managedItemList;
    static ADMINISTRATOR_ID                                 GlobalID; // a counter to assigned number
};

class ManagedItem
{
  protected:
    ManagedItem(const Administrator &admin);

  public:
    bool           checkPermission(const PermissionType perm) const;
    PermissionType getPermission() const;
    bool           isReadable() const;
    bool           isWritable() const;
    void           setPermission(const Administrator &admin, const PermissionType perm) const;
    void           addPermission(const Administrator &admin, const PermissionType perm) const;

  public:
    const ADMINISTRATOR_ID administrator_ID;

  protected:
    mutable PermissionType permission = lowest;
};
template <class T> class ManagedVal : public ManagedItem
{
  public:
    ManagedVal(const Administrator &admin);
    ManagedVal(const Administrator &admin, const T &val);

         operator const T();
    T    read() const;
    void write(const T &val);

  private:
    std::unique_ptr<T> value;
};
template <typename T> ManagedVal<T>::ManagedVal(const Administrator &admin) : ManagedItem(admin), value(nullptr){};
template <typename T> ManagedVal<T>::ManagedVal(const Administrator &admin, const T &val) : ManagedItem(admin)
{
    using namespace std;

    value = make_unique<T>(val);
}
template <typename T> ManagedVal<T>::operator const T()
{
    return read();
}
template <typename T> T ManagedVal<T>::read() const
{
    using namespace std;

    if (!this->checkPermission(readable)) throw runtime_error("Error: Insufficient permissions for read operation.");
    if (value)
        return *value;
    else
        return T();
}
template <typename T> void ManagedVal<T>::write(const T &val)
{
    using namespace std;

    if (!this->checkPermission(writable)) throw runtime_error("Error: Insufficient permissions for write operation.");
    value = make_unique<T>(val);
}

class ManagedClass
{
  protected:
    template <typename Y> void record(ManagedVal<Y> &managedVal, const Y &val) const;
    void                       refresh() const;
    template <typename Y>
    void copyManagedVal(ManagedVal<Y> &managedVal, const ManagedVal<Y> &otherManagedVal,
                        const ManagedClass &other) const;
    void copyManagedClass(const ManagedClass &other);

  protected:
    const Administrator administrator;
    mutable bool        isrefreshed = true;
};
template <typename Y> void ManagedClass::record(ManagedVal<Y> &managedVal, const Y &val) const
{
    managedVal.setPermission(administrator, writable);
    managedVal.write(val);
    managedVal.setPermission(administrator, readable);
    isrefreshed = false;
}
template <typename Y>
void ManagedClass::copyManagedVal(ManagedVal<Y> &managedVal, const ManagedVal<Y> &otherManagedVal,
                                  const ManagedClass &other) const
{
    PermissionType perm = otherManagedVal.getPermission();
    otherManagedVal.addPermission(other.administrator, readable);
    managedVal.addPermission(administrator, writable);
    managedVal.write(otherManagedVal.read());
    otherManagedVal.setPermission(other.administrator, perm);
    managedVal.setPermission(administrator, perm);
}
#endif // MANAGED_HPP
