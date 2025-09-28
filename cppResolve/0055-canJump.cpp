#include "common.h"

class Solution55 {
public:
    bool canJump(vector<int> &nums) {
        if (nums.size() == 1)
            return true;
        int min = 0;
        int max = nums[0];

        for (int i = min + 1; i <= max; i++) {
            if (max + 1 >= nums.size())
                return true;
            max = std::max(max, i + nums[i]);
        }

        return false;
    }
};