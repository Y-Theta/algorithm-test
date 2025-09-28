#include <algorithm>
#include <bitset>
#include <iostream>
#include <stack>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifndef USINGCOMMON
#define USINGCOMMON

#endif
using std::string;
using namespace std;

struct ListNode {
    int val;
    int key;
    ListNode *next = nullptr;
    ListNode *pre = nullptr;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

class TestSolution {
public:
    void Test() ;
};

class Solution3 {
public:
    int lengthOfLongestSubstring(string s);
};

class Solution128 {
public:
    int longestConsecutive(vector<int> &nums);
};

class Solution322 {
public:
    int coinChange(vector<int> &coins, int amount);
};

class Solution139 {
public:
    bool wordBreak(string s, vector<string> &wordDict);
};

class Solution279 {
public:
    int numSquares(int n);
};

class Solution300 {
public:
    int lengthOfLIS(vector<int> &nums);
};

class Solution152 {
public:
    int maxProduct(vector<int> &nums);
};

class Solution416 {
public:
    bool canPartition(vector<int> &nums);
};

class Solution560 {
public:
    int subarraySum(vector<int> &nums, int k);
};

class Solution11 {
public:
    int maxArea(vector<int> &height);
};

class Solution239 {
public:
    vector<int> maxSlidingWindow(vector<int> &height, int k);
};

class Solution73 {
public:
    void setZeroes(vector<vector<int>> &matrix);
};

class Solution48 {
public:
    void rotate(vector<vector<int>> &matrix);
};