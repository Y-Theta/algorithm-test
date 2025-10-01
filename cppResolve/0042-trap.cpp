#include "0-common.h"

class Solution42 {
public:
    int trap(vector<int> &height) {
        stack<int> buckets;
        for (int i = 0; i < height.size(); i++) {
            if (buckets.empty()) {
                buckets.push(height[i]);
            }
        }
    }
};