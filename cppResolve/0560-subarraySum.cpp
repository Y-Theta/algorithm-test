#include "common.h"

int Solution560::subarraySum(vector<int> &nums, int k) {
    // dp[i] = dp[i-1] + count(substr end with i)
    long sum[nums.size()];
    sum[0] = nums[0];
    for (int i = 1; i < nums.size(); i++) {
        sum[i] = sum[i - 1] + nums[i];
    }

    vector<int> dp(nums.size() + 1, 0);
    dp[0] = nums[0] == k ? 1 : 0;
    for (int i = 1; i < nums.size(); i++) {
        auto currentsum = sum[i];
        int endwithme = currentsum == k ? 1 : 0;
        for (int j = 0; j < i; j++) {
            if (currentsum - sum[j] == k)
                endwithme++;
        }

        dp[i] = dp[i - 1] + endwithme;
    }

    return dp[nums.size() - 1];
};