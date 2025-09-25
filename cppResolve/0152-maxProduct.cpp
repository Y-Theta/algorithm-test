#include "common.h"

typedef struct numinfo {
    int num1;
    int num2;
};

int Solution152::maxProduct(vector<int> &nums) {
    int summax = 0;
    numinfo numsinfo[nums.size()];
    numsinfo[0].num1 =
        numsinfo[0].num2 = nums[0];
    summax = nums[0];

    for (int i = 1; i < nums.size(); i++) {
        int val1 = nums[i];
        int val2 = nums[i] * numsinfo[i - 1].num1;
        int val3 = nums[i] * numsinfo[i - 1].num2;

        auto max = std::max(val1, std::max(val2, val3));
        numsinfo[i].num1 =
            numsinfo[i].num2 = max;
        summax = std::max(summax, max);
        auto min = std::min(val1, std::min(val2, val3));
        if (min < 0) {
            numsinfo[i].num2 = min;
        }
    }

    return summax;
};