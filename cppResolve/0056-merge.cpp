#include "0-common.h"

class Solution56 {
public:
    vector<vector<int>> merge(vector<vector<int>> &intervals) {
        // 1 1 1 1 1 0 0
        auto pt = intervals.data();
        std::sort(pt, pt + intervals.size(), [](vector<int> a, vector<int> b) -> int { return a[0] < b[0]; });

        for (int i = 0; i < intervals.size(); i++)
        {
           
        }
        
    }
};