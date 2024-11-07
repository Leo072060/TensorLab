#ifndef DECISION_TREE_HPP
#define DECISION_TREE_HPP

#include <memory>
#include <unordered_set>
#include <vector>

#include "ML/_internal/multiClassificationModelBase.hpp"

namespace TL
{
using namespace _internal;

enum SplitCriterion : int
{
    gain,
    gain_ratio,
    gini_index
};

class DecisionTree : public MultiClassificationModelBase<std::string>
{
  private:
    class TreeNode;
    Mat<double>               train_multi(const Mat<std::string> &x, const Mat<std::string> &y) override;
    Mat<std::string>          predict_multi(const Mat<std::string> &x, const Mat<double> &theta) const override;
    std::string               chooseSplitFeature(const Mat<std::string> &x, const Mat<std::string> &y,
                                                 const Mat<double> &weight) const;
    std::shared_ptr<TreeNode> generateTrees(const Mat<std::string> &x, const Mat<std::string> &y,
                                            const Mat<double> &weight) const;
    Mat<double>               tree2theta(std::shared_ptr<const TreeNode> tree) const;

    // for polymorphism
  public:
    std::shared_ptr<ClassificationModelBase<std::string>> clone() const override;

  private:
    class TreeNode
    {
      public:
        std::vector<std::pair<std::string, std::shared_ptr<TreeNode>>> ptr_subTrees;
        std::string feature_or_category;
        bool isLeaf = false;
    };

  public:
    SplitCriterion split_criterion = SplitCriterion::gain;
};
} // namespace TL

#endif // DECISION_TREE_HPP