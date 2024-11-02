#include "_internal/managed.hpp"

using namespace TL;
using namespace TL::_internal;

#pragma region Administrator
ADMINISTRATOR_ID Administrator::GlobalID = 0;
size_t           Administrator::managedItemsSize() const
{
    return managedItemList.size();
}
std::shared_ptr<ManagedItem> Administrator::getManagedItem(const size_t i) const
{
    return managedItemList[i];
}
void Administrator::registerManagedItem(std::shared_ptr<ManagedItem> ptr_ManagedItem)
{
    managedItemList.push_back(ptr_ManagedItem);
}
#pragma endregion

#pragma region ManagedItem
ManagedItem::ManagedItem(Administrator &admin) : administrator_ID(admin.ID), permission(Permission::lowest)
{
}
bool ManagedItem::checkPermission(const PermissionType perm) const
{
    return CheckFlag(permission, perm);
}
PermissionType ManagedItem::getPermission() const
{
    return permission;
}
bool ManagedItem::isReadable() const
{
    return checkPermission(Permission::readable);
}
bool ManagedItem::isWritable() const
{
    return checkPermission(Permission::writable);
}
void ManagedItem::setPermission(const Administrator &admin, const PermissionType perm)
{
    using namespace std;

    if (admin.ID != this->administrator_ID)
    {
        cerr << "Error: Permission denied: administrator ID mismatch. "
             << "Provided administrator ID: " << admin.ID << ", expected administrator ID: " << this->administrator_ID
             << "." << endl;
        throw runtime_error("Permission denied.");
    }

    permission = perm;
}
void ManagedItem::addPermission(const Administrator &admin, const PermissionType perm)
{
    using namespace std;

    if (admin.ID != this->administrator_ID)
    {
        cerr << "Error: Permission denied: administrator ID mismatch. "
             << "Provided administrator ID: " << admin.ID << ", expected administrator ID: " << this->administrator_ID
             << "." << endl;
        throw runtime_error("Permission denied.");
    }

    AddFlag(permission, perm);
}
void ManagedItem::delPermission(const Administrator &admin, const PermissionType perm)
{
    using namespace std;

    if (admin.ID != this->administrator_ID)
    {
        cerr << "Error: Permission denied: administrator ID mismatch. "
             << "Provided administrator ID: " << admin.ID << ", expected administrator ID: " << this->administrator_ID
             << "." << endl;
        throw runtime_error("Permission denied.");
    }

    DelFlag(permission, perm);
}
#pragma endregion

#pragma region ManagedClass
ManagedClass::ManagedClass(const ManagedClass &other) : administrator(), isrefreshed(other.isrefreshed) {}
ManagedClass::ManagedClass(ManagedClass &&other) noexcept : administrator(), isrefreshed(other.isrefreshed) {}
ManagedClass &ManagedClass::operator=(const ManagedClass &rhs)
{
    for (size_t i = 0; i < administrator.managedItemsSize(); ++i)
    {
        administrator.getManagedItem(i)->copy(administrator, rhs.administrator, *rhs.administrator.getManagedItem(i));
    }
    return *this;
}
ManagedClass &ManagedClass::operator=(ManagedClass &&rhs) noexcept
{
    for (size_t i = 0; i < administrator.managedItemsSize(); ++i)
    {
        administrator.getManagedItem(i)->copy(administrator, rhs.administrator, *rhs.administrator.getManagedItem(i));
    }
    return *this;
}
void ManagedClass::copyAfterConstructor(const ManagedClass &other)
{
    for (size_t i = 0; i < administrator.managedItemsSize(); ++i)
    {
        administrator.getManagedItem(i)->copy(administrator, other.administrator,
                                              *other.administrator.getManagedItem(i));
    }
}
void ManagedClass::refresh() const
{
    if (isrefreshed) return;
    for (size_t i = 0; i < administrator.managedItemsSize(); ++i)
        administrator.getManagedItem(i)->setPermission(administrator, lowest);
    isrefreshed = true;
}
#pragma endregion
