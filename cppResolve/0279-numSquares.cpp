#include "common.h"

int Solution279::numSquares(int n) {
    vector<int> nums;
    for (int i = 1; i * i <= n; ++i) {
        nums.push_back(i * i);
    }

    vector<int> dp(n + 1, n + 1);
    dp[0] = 0;
    dp[1] = 1;
    for (int i = 2; i <= n; i++) {
        for (int j = 0; j < nums.size(); j++) {
            if (i - nums[j] < 0)
                continue;
            dp[i] = std::min(dp[i], dp[i - nums[j]] + 1);
        }
    }

    return dp[n];
};