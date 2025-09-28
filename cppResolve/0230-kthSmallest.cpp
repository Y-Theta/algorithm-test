#include "common.h"

class Solution230 {
public:
    int i = 0;
    int kval = 0;
    int aim = 0;
    int kthSmallest(TreeNode *root, int k) {
        aim = k;
        traversal(root);
        return kval;
    }

    void traversal(TreeNode *root) {
        if (root == nullptr)
            return;
        if (root->left != nullptr) {
            traversal(root->left);
        }
        i++;
        if (i == aim) {
            kval = root->val;
        }
        if (root->right != nullptr) {
            traversal(root->right);
        }
    }
};