#include "common.h"

class Solution169 {
public:
    int majorityElement(vector<int> &nums) {
        int num = 1;
        int current = nums[0];
        for (size_t i = 1; i < nums.size(); i++) {
            if (num == 0) {
                current = nums[i];
            }
            if (current == nums[i]) {
                num++;
            } else {
                num--;
            }
        }
        return current;
    }
};