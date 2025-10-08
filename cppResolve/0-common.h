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
#include <queue>

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
    void Test();
};

class SolutionTest1 {
public:
    int lengthOfLongestSubstring(string s);
    int maxArea(vector<int> &height);
    void rotate(vector<vector<int>> &matrix);
    void setZeroes(vector<vector<int>> &matrix);
    int longestConsecutive(vector<int> &nums);
    bool wordBreak(string s, vector<string> &wordDict);
    int maxProduct(vector<int> &nums);
    vector<int> maxSlidingWindow(vector<int> &nums, int k);
    int numSquares(int n);
    int lengthOfLIS(vector<int> &nums);
    int coinChange(vector<int> &coins, int amount) ;
    bool canPartition(vector<int> &nums);
    int subarraySum(vector<int> &nums, int k);
    string longestPalindrome(string s);
    int trap(vector<int> &height);
    int longestCommonSubsequence(string text1, string text2);
    vector<int> successfulPairs(vector<int>& spells, vector<int>& potions, long long success);
};
