#include "common.h"

int Solution128::longestConsecutive(vector<int> &nums) {
    if (nums.size() == 0)
        return 0;
    if (nums.size() == 1)
        return 1;

    int *list = nums.data();
    sort(list, list + nums.size());

    int max = 0;
    int start = 0;
    int end = 0;
    for (int i = 1; i < nums.size(); i++) {
        if (nums[i] - nums[i - 1] == 1) {
            end++;
            continue;
        }
        else if(nums[i] - nums[i - 1] == 0){
            end++;
            start++;
            continue;
        }
         else {
            max = std::max(max, end - start + 1);
            start = i;
            end = i;
        }
    }
    max = std::max(max, end - start + 1);
    return max;
};