#include "common.h"

class Solution35 {
public:
    int searchInsert(vector<int> &nums, int target) {
        return search(nums, target, 0, nums.size() - 1);
    }

    int search(vector<int> &nums, int target, int start, int end) {
        int mid = (start + end) >> 1;
        if (start == end) {
            if (nums[mid] < target) {
                return start + 1;
            } else {
                return start;
            }
        }

        if (nums[mid] > target) {
            return search(nums, target, 0, mid);
        } else if (nums[mid] < target) {
            return search(nums, target, mid + 1, end);
        } else {
            return mid;
        }
    }
};