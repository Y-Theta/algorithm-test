#include "common.h"

class Solution121 {
public:
    int maxProfit(vector<int> &prices) {
        int min = prices[0];
        int max = prices[0];

        int maxprofit = 0;

        for (int i = 1; i < prices.size(); i++) {
            if (prices[i] < min) {
                maxprofit = std::max(maxprofit, max - min);
                min = prices[i];
                max = prices[i];
            }
            if (prices[i] > max) {
                max = prices[i];
                maxprofit = std::max(maxprofit, max - min);
            }
        }

        return maxprofit;
    }
};