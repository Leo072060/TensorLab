#include "kits/managed.hpp"

using namespace std;

ADMINISTRATOR_ID Administrator::GlobalID = 0;

size_t Administrator::constManagedItemsSize() const
{
    return constManagedItemList.size();
}
size_t Administrator::managedItemsSize() const
{
    return managedItemList.size();
}
const ManagedItem &Administrator::getConstManagedItem(const size_t i) const
{
    return *constManagedItemList[i];
}
ManagedItem &Administrator::getManagedItem(const size_t i) const
{
    return *managedItemList[i];
}
void Administrator::registerConstManagedItem(const ManagedItem &managedItem) const
{
    constManagedItemList.push_back(make_unique<const ManagedItem>(managedItem));
}
void Administrator::registerManagedItem(ManagedItem &managedItem) const
{
    managedItemList.push_back(make_unique<ManagedItem>(managedItem));
}

ManagedItem::ManagedItem(const Administrator &admin) : administrator_ID(admin.ID)
{
    admin.registerManagedItem(*this);
}
bool ManagedItem::checkPermission(const Permission perm) const
{
    return CHECK_FLAG(permission, perm);
}
PermissionType ManagedItem::getPermission() const
{
    return permission;
}
bool ManagedItem::isReadable() const
{
    return checkPermission(readable);
}
bool ManagedItem::isWritable() const
{
    return checkPermission(writable);
}
void ManagedItem::setPermission(const Administrator &admin, const Permission perm) const
{
    if (admin.ID != this->administrator_ID)
        throw runtime_error("Error: This administrator does not have permission to modify.");
    permission = perm;
}
void ManagedItem::addPermission(const Administrator &admin, const Permission perm) const
{
    if (admin.ID != this->administrator_ID)
        throw runtime_error("Error: This administrator does not have permission to modify.");
    if (this->checkPermission(perm)) return;
    permission = permission & perm;
}

void ManagedClass::refresh() const
{
    if (isrefreshed) return;
    for (size_t i = 0; i < administrator.constManagedItemsSize(); ++i)
    {
        administrator.getConstManagedItem(i).setPermission(administrator, lowest);
    }
    for (size_t i = 0; i < administrator.managedItemsSize(); ++i)
    {
        administrator.getManagedItem(i).setPermission(administrator, lowest);
    }
    isrefreshed = true;
}