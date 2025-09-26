#include "common.h"

bool Solution416::canPartition(vector<int> &nums) {
    if (nums.size() == 1)
        return 0;
    vector<int> numscount(101, 0);
    int total = 0;
    for (int i = 0; i < nums.size(); i++) {
        total += nums[i];
        numscount[nums[i]]++;
    }
    if (total % 2 != 0)
        return 0;

    auto aim = total / 2;
    if (aim <= 100 && numscount[aim] > 0)
        return true;

    unordered_set<int> currentsums;
    currentsums.insert(0);
    vector<int> currentnew;
    vector<int> numsoccured(101, 0);
    for (int i = 0; i < nums.size(); i++) {
        if (numsoccured[nums[i]] > 0)
            return true;
        currentnew.clear();
        for (int j : currentsums) {
            auto remain = j - nums[i];
            if (remain > 0) {
                currentnew.push_back(remain);
                if (remain <= 100 && remain >= 0) {
                    numsoccured[remain] = 1;
                }
            }
        }
        for (int k = 0; k < currentnew.size(); k++) {
            currentsums.insert(currentnew[k]);
        }
        currentsums.insert(aim - nums[i]);
        if (aim - nums[i] <= 100 && aim - nums[i] >= 0) {
            numsoccured[aim - nums[i]] = 1;
        }
    }

    return false;
};