#include "0-common.h"

class Solution1 {
public:
    vector<int> twoSum(vector<int> &nums, int target) {
        unordered_map<int,int> dic;
        for (int i = 0; i < nums.size(); i++)
        {
            auto dec = target - nums[i];
            bool exist = dic.count(dec) > 0;
            if (exist)
            {
                return vector<int>{dic[dec], i};
            }
            dic[nums[i]] = i;
        }
        return vector<int>(0);
    }
};