#include "0-common.h"

class Solution94 {
public:
    vector<int> inorderTraversal(TreeNode *root) {
        vector<int> x;
        inorderTraversa(root, x);
        return x;
    }

    void inorderTraversa(TreeNode *root, vector<int> &x) {
        if (root == nullptr)
            return;
        if (root->left != nullptr) {
            inorderTraversa(root->left, x);
        }
        x.push_back(root->val);
        if (root->right != nullptr) {
            inorderTraversa(root->right, x);
        }
    }
};