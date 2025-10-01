#include "0-common.h"

class Solution136 {
public:
    int singleNumber(vector<int>& nums) {
        int num = nums[0];
        for (size_t i = 1; i < nums.size(); i++)
        {
            num ^= nums[i];
        }
        
        return num;
    }
};