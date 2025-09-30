#include "common.h"

vector<int> Solution239::maxSlidingWindow(vector<int> &nums, int k) {
    if (k == 1 || nums.size() == 1)
        return nums;
    vector<int> newv;
    // 0  1 2  3 4 5 6
    vector<int> queue;
    int i = 0;
    for (; i < nums.size(); i++) {
        if (i >= k) {
            newv.push_back(queue[0]);
        }
        if (i - k >= 0) {
            if (nums[i - k] == queue[0]) {
                queue.erase(queue.begin());
            }
        }
        if (queue.empty() || nums[i] <= queue[queue.size() - 1]) {
            queue.push_back(nums[i]);
        } else  {
            while (!queue.empty() && queue[queue.size() - 1] < nums[i]) {
                    queue.pop_back();
            }
            queue.push_back(nums[i]);
        }
    }
    newv.push_back(queue[0]);

    return newv;
};