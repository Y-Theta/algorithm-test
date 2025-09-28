#include "common.h"

class Solution53 {
public:
    int maxSubArray(vector<int> &nums) {
        // dp[i] = max( dp[i-1],nums[i-1])
        if (nums.size() == 1)
            return nums[0];
        int dp[nums.size()];
        dp[0] = nums[0];
        int max =  dp[0] ;
        for (int i = 1; i < nums.size(); i++) {
            dp[i] = std::max(dp[i - 1] + nums[i], nums[i]);
            max = std::max(dp[i], max);
        }
        return max;
    }
};