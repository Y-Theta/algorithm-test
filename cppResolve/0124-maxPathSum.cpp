#include "0-common.h"

class Solution124 {
public:
    int summax = 0;
    int maxPathSum(TreeNode *root) {
        // node is selected only if sum > node
        // everynode max = left max || right max || left + right max
        summax = root->val;
        getmax(root);
        return summax;
    }

    int getmax(TreeNode *root) {
        summax = std::max(summax, root->val);
        int leftmax = -10000000, rightmax = -10000000;
        if (root->left == nullptr && root->right == nullptr)
            return root->val;

        if (root->left != nullptr) {
            leftmax = getmax(root->left) + root->val;
        }
        if (root->right != nullptr) {
            rightmax = getmax(root->right) + root->val;
        }

        auto openmax = std::max(root->val, std::max(leftmax, rightmax));
        summax = std::max(summax, std::max(openmax, rightmax + leftmax - root->val));
        return openmax;
    }
};