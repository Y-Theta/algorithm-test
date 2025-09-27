#include "common.h"

typedef struct posval {
    int val;
    int pos;
};

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

// vector<int> Solution239::maxSlidingWindow(vector<int> &nums, int k) {
//     if (k == 1 || nums.size() == 1)
//         return nums;

//     vector<int> newv(nums.size() - k + 1, -100001);
//     for (int i = 0; i < nums.size(); i++) {
//         int max = std::min((int)newv.size() - 1, i);
//         int min = std::max(0, i - k + 1);
//         for (int j = max; j >= min; j--) {
//             if (nums[i] > newv[j]) {
//                 newv[j] = nums[i];
//             }
//         }
//     }

//     return newv;
// };