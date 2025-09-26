#include "common.h"

class Solution75 {
public:
    void sortColors(vector<int> &nums) {
        int count[3];
        for (int i = 0; i < nums.size(); i++)
        {
            count[nums[i]]++;
        }
        
        int k = 0;
        for (int i = 0; i < sizeof(count) / sizeof(int); i++)
        {
            for (int j = 0; j < count[i]; j++)
            {
                nums[k++] = i;
            }
        }
    }
};