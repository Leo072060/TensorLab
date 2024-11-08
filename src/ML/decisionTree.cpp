#include <cmath>
#include <functional>
#include <map>
#include <unordered_map>

#include "ML/decisionTree.hpp"
#include "computation/entropy.hpp"

using namespace TL;

Mat<double> DecisionTree::train_multi(const Mat<std::string> &x, const Mat<std::string> &y)
{
    using namespace std;
    Mat<double> weight(x.size(Axis::row), 1);
    weight                                       = 1;
    std::shared_ptr<DecisionTree::TreeNode> root = generateTrees(x, y, weight);
    return tree2theta(root);
}
Mat<std::string> DecisionTree::predict_multi(const Mat<std::string> &x, const Mat<double> &theta) const
{
    using namespace std;

    Mat<std::string> y(x.size(Axis::row), 1);
    for (size_t r = 0; r < x.size(Axis::row); ++r)
    {
        size_t cur = 0;
        while (cur < theta.size(Axis::row))
        {
            if (theta.iloc(cur, 0) < 0) // is not a leaf
            {
                size_t index_feature = x.name2index(theta.iloc_name(cur, Axis::row), Axis::col);
                if (x.iloc(r, index_feature).empty())
                {
                    cur = cur + 1;
                }
                else
                {
                    size_t i = cur + 1;
                    for (; i < cur - theta.iloc(cur, 0) + 1; ++i)
                    {
                        if (theta.iloc_name(i, Axis::row) == x.iloc(r, index_feature))
                        {
                            cur = i;
                            break;
                        }
                    }
                    if (i != cur)
                    {
                        cerr << "Unknown attribute encountered: Feature: " << theta.iloc_name(cur, Axis::row)
                             << ", Attribute: " << x.iloc(r, index_feature) << ", in the line of theta: " << cur
                             << endl;
                        throw runtime_error("Unknown attribute.");
                    }
                    cur = theta.iloc(cur, 0);
                }
            }
            else if (0 == theta.iloc(cur, 0)) // is a leaf
            {
                y.iloc(r, 0) = theta.iloc_name(cur, Axis::row);
                break;
            }
            else
            {
                cerr << "Error: The theta is wrong." << endl;
                throw invalid_argument("Wrong theta.");
            }
        }
        if (cur >= theta.size(Axis::row))
        {
            cerr << "Error: The theta is wrong." << endl;
            throw invalid_argument("Wrong theta.");
        }
    }

    return y;
}
std::string DecisionTree::chooseSplitFeature(const Mat<std::string> &x, const Mat<std::string> &y,
                                             const Mat<double> &weight) const
{
    using namespace std;

    switch (split_criterion)
    {
    case gain: {
        double max_ent_inc  = -1;
        size_t splitFeature = 0;
        for (size_t c = 0; c < x.size(Axis::col); ++c)
        {
            Mat<string> attributes = x.unique();
            double      ent_inc    = 0;
            for (size_t i = 0; i < attributes.size(); ++i)
            {
                Mat<string> y_i;
                Mat<string> weight_i;
                for (size_t r = 0; r < x.size(Axis::row); ++r)
                {
                    if (x.iloc(r, c) == attributes.iloc(0, i))
                    {
                        y_i.concat(y.iloc(r, Axis::row), Axis::row);
                        weight_i.concat(weight.iloc(r, Axis::row), Axis::row);
                    }
                }
                ent_inc -= entropy(y_i, weight_i);
            }
            if (ent_inc > max_ent_inc)
            {
                max_ent_inc  = ent_inc;
                splitFeature = c;
            }
        }
        return x.iloc_name(splitFeature, Axis::col);
    }
    case gain_ratio:
    case gini_index:
    default:
        cerr << "Error: Invalid split criterion." << endl;
        throw runtime_error("Invalid split criterion.");
    }
}
std::shared_ptr<DecisionTree::TreeNode> DecisionTree::generateTrees(const Mat<std::string> &x,
                                                                    const Mat<std::string> &y,
                                                                    const Mat<double>      &weight) const
{
    using namespace std;
    shared_ptr<TreeNode> root = make_shared<TreeNode>();

    // Here are two cases fot generating leaf node.
    // Case 1: All samples belong to one type.
    if (1 == y.unique().size())
    {
        root->isLeaf              = true;
        root->feature_or_category = y.unique().iloc(0, 0);
        return root;
    }
    // Case 2: The number of available features is 0, or the values of the available features are the same.
    bool areFeatureValuesIdentical = true;
    for (size_t c = 0; c < x.size(Axis::col); ++c)
    {
        if (x.iloc(c, Axis::col).unique().size() > 1)
        {
            areFeatureValuesIdentical = false;
            break;
        }
    }
    if (areFeatureValuesIdentical)
    {
        unordered_map<string, size_t> statistic = y.count();
        string                        category;
        size_t                        count = 0;
        for (const auto e : statistic)
            if (e.second > count)
            {
                category = e.first;
                count    = e.second;
            }
        root->isLeaf              = true;
        root->feature_or_category = y.unique().iloc(0, 0);
        root->feature_or_category = category;
        return root;
    }

    // Generate the subtrees.
    string splitFeature       = chooseSplitFeature(x, y, weight);
    root->feature_or_category = splitFeature;

    Mat<string> x_rm_splitFeature  = x;
    size_t      index_splitFeature = x.name2index(splitFeature, Axis::col);
    x_rm_splitFeature.drop(index_splitFeature, Axis::col);

    multimap<double, std::pair<std::string, std::shared_ptr<TreeNode>>, std::greater<double>> subTrees;
    Mat<string> attributes = x.iloc(index_splitFeature, Axis::col).unique();

    if (0 == x_rm_splitFeature.size())
    {
        double      weight_attribute = 0;
        Mat<string> newY;
        for (size_t i = 0; i < attributes.size(); ++i)
        {
            for (size_t r = 0; r < x.size(Axis::row); ++r)
            {
                if (x.iloc(r, index_splitFeature) == attributes.iloc(0, i))
                {
                    newY = newY.concat(y.iloc(r, Axis::row), Axis::row);
                    weight_attribute += weight.iloc(r, 0);
                }
            }

            // Find the category with the highest count.
            unordered_map<string, size_t> statistic = newY.count();
            string                        maxCategory;
            size_t                        maxCount = 0;
            for (const auto &e : statistic)
                if (e.second > maxCount)
                {
                    maxCount    = e.second;
                    maxCategory = e.first;
                }

            shared_ptr<TreeNode> subTree = make_shared<TreeNode>();
            subTree->isLeaf              = true;
            subTree->feature_or_category = maxCategory;
            subTrees.insert({weight_attribute, {attributes.iloc(0, i), subTree}});
        }
    }
    else
    {
        double         totalWeight = 0;  // sum weight of x which the 'splitFeature' is not missing
        vector<size_t> index_missingVal; // collect index of missing value
        for (size_t r = 0; r < x.size(Axis::row); ++r)
        {
            if (!x.iloc(r, index_splitFeature).empty())
                ++totalWeight;
            else
                index_missingVal.push_back(r);
        }

        size_t count_missingVal = 0; // record the number of missing values that have already been assigned.
        for (size_t i = 0; i < attributes.size(); ++i)
        {
            Mat<string> newX;
            Mat<string> newY;
            Mat<double> newWeight;
            double      weight_attribute = 0;
            for (size_t r = 0; r < x.size(Axis::row); ++r)
            {
                if (x.iloc(r, index_splitFeature) == attributes.iloc(0, i))
                {
                    newX      = newX.concat(x_rm_splitFeature.iloc(r, Axis::row), Axis::row);
                    newY      = newY.concat(y.iloc(r, Axis::row), Axis::row);
                    newWeight = newWeight.concat(weight.iloc(r, Axis::row), Axis::row);
                    weight_attribute += weight.iloc(r, 0);
                }
            }
            for (size_t j = 0; j < weight_attribute / totalWeight && count_missingVal < index_missingVal.size(); ++j)
            {
                newX = newX.concat(x_rm_splitFeature.iloc(index_missingVal[count_missingVal], Axis::row), Axis::row);
                newY = newY.concat(y.iloc(index_missingVal[count_missingVal], Axis::row), Axis::row);
                newWeight = newWeight.concat(weight.iloc(index_missingVal[count_missingVal], Axis::row), Axis::row);
                ++count_missingVal;
            }
            subTrees.insert({weight_attribute, {attributes.iloc(0, i), generateTrees(newX, newY, newWeight)}});
        }
    }
    for (const auto e : subTrees)
    {
        root->ptr_subTrees.push_back(e.second);
    }

    return root;
}
Mat<double> DecisionTree::tree2theta(std::shared_ptr<const TreeNode> tree) const
{
    /*
    _____________________________
    0   |feature1       |   -3  | <-- (-1- means it has 3(|-3|) attributes  feature1
    1   |attribute1     |   4   | <-- 4 means jump to location 4 if the     |
    2   |attribute2     |   9   |     value feature1 is attribute1          |_attribute1_feature2
    3   |attribute3     |   10  |                                           |            |
    4   |feature2       |   -2  |                                           |            |_attribute4_category1
    5   |attribute4     |   7   |                                           |            |
    6   |attribute5     |   8   |                                           |            |_attribute5_category2
    7   |category1      |   0   | <-- 0 means it is leaf                    |
    8   |category2      |   0   |                                           |_attribute2_category3
    9   |category3      |   0   |                                           |
    10  |category4      |   0   |                                           |_attribute3_category4
    ____|_______________|_______|  
    */
    using namespace std;

    vector<string>                  rowNames;
    vector<int>                     jumpTable;
    std::shared_ptr<const TreeNode> cur_tree  = tree;
    size_t                          cur       = 0;
    std::function<void()>           traversal = [&]() {
        if (cur_tree->isLeaf)
        {
            jumpTable.emplace_back(0);
            rowNames.emplace_back(cur_tree->feature_or_category);
            return;
        }
        else
        {
            jumpTable.emplace_back(-cur_tree->ptr_subTrees.size());
            rowNames.emplace_back(cur_tree->feature_or_category);
            size_t jumpNum = jumpTable.size();
            for (const auto e : cur_tree->ptr_subTrees)
            {
                jumpTable.emplace_back(0);
                rowNames.push_back(e.first);
            }
            std::shared_ptr<const TreeNode> cur_origin = cur_tree;
            for (const auto e : cur_origin->ptr_subTrees)
            {
                jumpTable[jumpNum] = jumpTable.size();
                ++jumpNum;
                cur_tree = e.second;
                traversal();
            }
        }
    };
    traversal();
    Mat<double> ret(jumpTable.size(), 1);
    for (size_t r = 0; r < jumpTable.size(); ++r)
    {
        ret.iloc(r, 0)              = jumpTable[r];
        ret.iloc_name(r, Axis::row) = rowNames[r];
    }

    return ret;
}

std::shared_ptr<ClassificationModelBase<std::string>> DecisionTree::clone() const
{
    using namespace std;

    return make_shared<DecisionTree>(*this);
}