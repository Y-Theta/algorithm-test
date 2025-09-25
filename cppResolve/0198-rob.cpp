#include "common.h"

class Solution198 {
public:
    int rob(vector<int> &nums) {
        // dp[i] = max( dp[i - 2] ,dp[i  - 3]) + nums[i];
        int totalmax = 0;
        if (nums.size() == 1)
            return nums[0];
        vector<int> rob(nums.size() + 1, 0);
        rob[0] = nums[0];
        rob[1] = nums[1];
        for (int i = 2; i < nums.size(); i++) {
            int currentmax = 0;
            if (i - 2 >= 0) {
                currentmax = std::max(currentmax, rob[i - 2]);
            }
            if (i - 3 >= 0) {
                currentmax = std::max(currentmax, rob[i - 3]);
            }
            rob[i] = currentmax + nums[i];
        }

        return *std::max_element(rob.begin(), rob.end());
    }
};