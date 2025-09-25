#include "common.h"

class Solution322 {
public:
    int coinChange(vector<int> &coins, int amount) {
        // dp[i] = i % k == 0? i / k :math.min(i / k + dp[i % k],);
        if (amount == 0)
            return 0;

        int *coinspt = coins.data();
        sort(coinspt, coinspt + coins.size(), [](int a, int b) -> int { b - a; });

        if (amount < coins[coins.size() - 1]) {
            return -1;
        }

    }
};