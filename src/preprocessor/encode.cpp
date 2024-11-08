#include "preprocessor/encode.hpp"

namespace TL
{
Mat<int> onehot_encode(Mat<std::string> data)
{
    using namespace std;

    if (0 == data.size()) return Mat<int>();

    Mat<string> uniqueData = data.unique();
    Mat<int>    ret(data.size(Axis::row), uniqueData.size());

    for (size_t r = 0; r < data.size(Axis::row); ++r)
    {
        size_t onehot       = uniqueData.find(data.iloc(r, 0)).at(0).second;
        ret.iloc(r, onehot) = 1;
    }

    return ret;
}
} // namespace TL
