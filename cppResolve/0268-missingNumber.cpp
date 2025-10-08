#include "0-common.h"

class Solution268 {
public:
    int missingNumber(vector<int> &nums) {
        std::sort(nums.begin(), nums.end());
        if (nums.size() == 1) {
            return nums[0] == 0 ? 1 : 0;
        }
        if (nums[0] != 0)
            return 0;
        for (int i = 1; i < nums.size(); i++) {
            if (nums[i] - nums[i - 1] > 1) {
                return nums[i - 1] + 1;
            }
        }
        return nums.size();
    }
};