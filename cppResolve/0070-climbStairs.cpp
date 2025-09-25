#include "common.h"

class Solution70 {
public:
    int climbStairs(int n) {
        // dp[n] = (dp[n-1] + 1) + (dp[n-2] + 1)
        if (n == 1) {
            return 1;
        }
        long dp[n + 1];
        dp[1] = 2;
        dp[2] = 3;
        for (size_t i = 3; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2] ;
        }

        return dp[n - 1];
    }
};