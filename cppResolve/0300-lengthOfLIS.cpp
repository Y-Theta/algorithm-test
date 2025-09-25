#include "common.h"

int Solution300::lengthOfLIS(vector<int> &nums) {
    // 10 11 12 13 14 2 3 4 5 6 7 8 9
    int max = 1;
    vector<int> dp(nums.size(), 1);

    for (int i = 1; i < nums.size(); i++) {
        for (int j = 0; j < i; j++) {
            if (nums[j] < nums[i]) {
                dp[i] = std::max(dp[i], dp[j] + 1);
                max = std::max(dp[i], max);
            }
        }
    }

    return max;
};