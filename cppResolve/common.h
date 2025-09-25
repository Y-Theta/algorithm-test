#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>

using std::string;
using namespace std;

struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

class Solution3 {
public:
    int lengthOfLongestSubstring(string s);
};

class Solution128 {
public:
    int longestConsecutive(vector<int>& nums);
};

class Solution322 {
public:
    int coinChange(vector<int> &coins, int amount);
};

class Solution139{
public:
bool wordBreak(string s, vector<string> &wordDict);
};

class Solution279{
public:
int numSquares(int n);
};