#include "0-common.h"

int SolutionTest1::coinChange(vector<int> &coins, int amount) {
    // dp[i] = i % k == 0? i / k :math.min(i / k + dp[i % k], (i/ k - 1) + dp[k + i % k]);
    if (amount == 0)
        return 0;

    int coinssize = coins.size();

    int Max = amount + 1;
    vector<int> dp(amount + 1, Max);
    dp[0] = 0;
    for (int i = 1; i <= amount; i++) {
        for (int j = 0; j < coinssize; j++) {
            if (i - coins[j] < 0)
                continue;
            dp[i] = std::min(dp[i], dp[(int)(i - coins[j])] + 1);
        }
    }

    return dp[amount] >= Max ? -1 : dp[amount];
};