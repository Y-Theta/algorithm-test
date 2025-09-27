#include "common.h"

class Solution283 {
public:
    void moveZeroes(vector<int> &nums) {
        // 1 0 2 0 3
        int zeroCount = 0;

        for (int i = 0; i < nums.size(); i++) {
            if (i + zeroCount >= nums.size()) {
                for (int j = i; j < nums.size(); j++) {
                    nums[j] = 0;
                }
                return;
            }
            nums[i] = nums[i + zeroCount];
            if (nums[i] == 0) {
                while (nums[i] == 0) {
                    zeroCount++;
                    if (i + zeroCount < nums.size()) {
                        nums[i] = nums[i + zeroCount];
                    } else {
                        for (int j = i + 1; j < nums.size(); j++) {
                            nums[j] = 0;
                        }
                        return;
                    }
                }
            }
        }
    }
};