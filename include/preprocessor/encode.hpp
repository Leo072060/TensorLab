#ifndef ENCODE_HPP
#define ENCODE_HPP

#include "mat/mat.hpp"

namespace TL
{
using namespace _internal;

Mat<int> onehot_encode(Mat<std::string> data);
} // namespace TL

#endif // ENCODE_HPP