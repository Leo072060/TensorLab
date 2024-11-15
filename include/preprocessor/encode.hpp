#ifndef ENCODE_HPP
#define ENCODE_HPP

#include "mat/mat.hpp"

namespace TL
{
using namespace _internal;

class OneHotEncoder : public ManagedClass
{
  public:
    OneHotEncoder();
    OneHotEncoder(const OneHotEncoder &other);
    OneHotEncoder(OneHotEncoder &&other) noexcept;
    OneHotEncoder &operator=(const OneHotEncoder &rhs);
    OneHotEncoder &operator=(OneHotEncoder &&rhs) noexcept;
    ~OneHotEncoder() = default;

    void             set_labels(const Mat<std::string> &labels);
    Mat<std::string> get_labels() const;
    void             fit(const Mat<std::string> &data);
    Mat<int>         transform(const Mat<std::string> &data) const;
    Mat<int>         fit_transform(const Mat<std::string> &data);
    Mat<std::string> decode(const Mat<int> &code) const;

  private:
    ManagedVal<Mat<std::string>> managed_labels;
};
} // namespace TL

#endif // ENCODE_HPP