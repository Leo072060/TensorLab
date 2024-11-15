#include "preprocessor/encode.hpp"

using namespace TL;

OneHotEncoder::OneHotEncoder()
    : ManagedClass()
    , managed_labels(this->administrator)
{
}
OneHotEncoder::OneHotEncoder(const OneHotEncoder &other)
    : ManagedClass(other)
    , managed_labels(this->administrator, other.administrator, other.managed_labels)
{
}
OneHotEncoder::OneHotEncoder(OneHotEncoder &&other) noexcept
    : ManagedClass(std::move(other))
    , managed_labels(this->administrator, other.administrator, other.managed_labels)
{
}
OneHotEncoder &OneHotEncoder::operator=(const OneHotEncoder &rhs)
{
    using namespace std;
    ManagedClass::operator=(rhs);
    managed_labels.copy(this->administrator, rhs.administrator, rhs.managed_labels);
    return *this;
}
OneHotEncoder &OneHotEncoder::operator=(OneHotEncoder &&rhs) noexcept
{
    using namespace std;
    ManagedClass::operator=(move(rhs));
    managed_labels.copy(this->administrator, rhs.administrator, rhs.managed_labels);
    return *this;
}
void OneHotEncoder::set_labels(const Mat<std::string> &labels)
{
    record(managed_labels, labels);
}
Mat<std::string> OneHotEncoder::get_labels() const
{
    using namespace std;

    if (!managed_labels.isReadable())
    {
        cerr << "Error: Labels are not set in the encoder." << endl;
        throw runtime_error("Fit or set labels first.");
    }

    return managed_labels.read();
}
void OneHotEncoder::fit(const Mat<std::string> &data)
{
    using namespace std;

    Mat<string> labels = data.unique();
    record(managed_labels, labels);
}
Mat<int> OneHotEncoder::transform(const Mat<std::string> &data) const
{
    using namespace std;

    if (!managed_labels.isReadable())
    {
        cerr << "Error: Labels are not set in the encoder." << endl;
        throw runtime_error("Fit or set labels first.");
    }

    Mat<int> ret(data.size(Axis::row), managed_labels.read().size(Axis::col));

    for (size_t r = 0; r < data.size(Axis::row); ++r)
    {
        size_t oneHot       = managed_labels.read().find(data.iloc(r, 0)).at(0).second;
        ret.iloc(r, oneHot) = 1;
    }

    return ret;
}
Mat<int> OneHotEncoder::fit_transform(const Mat<std::string> &data)
{
    fit(data);
    return transform(data);
}
Mat<std::string> OneHotEncoder::decode(const Mat<int> &code) const
{
    using namespace std;

    if (!managed_labels.isReadable())
    {
        cerr << "Error: Labels are not set in the encoder." << endl;
        throw runtime_error("Fit or set labels first.");
    }

    if (code.size(Axis::col) != managed_labels.read().size(Axis::col))
    {
        cerr << "Error: The number of columns in code does not match the labels." << endl;
        throw runtime_error("Dimension mismatch between code and labels.");
    }

    Mat<string> decoded(code.size(Axis::row), 1);

    for (size_t r = 0; r < code.size(Axis::row); ++r)
    {
        bool found = false;
        for (size_t c = 0; c < code.size(Axis::col); ++c)
        {
            if (code.iloc(r, c) == 1)
            {
                decoded.iloc(r, 0) = managed_labels.read().iloc(0, c);
                found              = true;
                break;
            }
        }
        if (!found)
        {
            cerr << "Error: Invalid one-hot encoding in row " << r << ": no '1' found." << endl;
            throw runtime_error("Invalid one-hot encoding in input code.");
        }
    }

    return decoded;
}
