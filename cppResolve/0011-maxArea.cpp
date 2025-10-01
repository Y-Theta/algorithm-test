#include "0-common.h"

int SolutionTest1::maxArea(vector<int> &height) {
    // val[i,j] = min(i,j) * (j - i)
    // left height right height
    if (height.size() == 1)
        return 0;
    int left = 0;
    int right = height.size() - 1;

    auto calc = [](vector<int> &height, int start, int end) -> int {
        return std::min(height[start], height[end]) * (end - start);
    };

    int max = calc(height, 0, height.size() - 1);
    for (int i = 0; i < height.size(); i++) {
        if (height[i] >= height[left]) {
            left = i;
            for (int j = right; j >= left; j--) {
                if (height[j] >= height[right]) {
                    right = j;
                    max = std::max(max, calc(height, left, j));
                }
            }
            right = height.size() - 1;
        }
    }

    return max;
};