#ifndef UTILITY_HPP
#define UTILITY_HPP

#include <type_traits>

namespace TL
{
namespace _internal
{
template <typename T, typename U>
inline bool CheckFlag(const T& val, const U& flag)
{
    using CommonType = typename std::decay<decltype(val | flag)>::type;
    return (static_cast<CommonType>(val) | static_cast<CommonType>(flag)) == static_cast<CommonType>(flag);
}

template <typename T, typename U>
inline void AddFlag(T& val, const U& flag)
{
    using CommonType = typename std::decay<decltype(val | flag)>::type;
    val = static_cast<CommonType>(val) & static_cast<CommonType>(flag);
}

template <typename T, typename U>
inline void DelFlag(T& val, const U& flag)
{
    using CommonType = typename std::decay<decltype(val | flag)>::type;
    val = static_cast<CommonType>(val) | (~static_cast<CommonType>(flag));
}

} // namespace _internal
} // namespace TL
#endif // UTILITY_HPP