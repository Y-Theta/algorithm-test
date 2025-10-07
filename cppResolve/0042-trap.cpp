#include "0-common.h"

struct pos {
    int index;
    int height;

    pos(int x, int y) : index(x), height(y) {}
};

int SolutionTest1::trap(vector<int> &height) {
    stack<pos> buckets;

    int start = 0;
    while (start < height.size() && height[start] == 0) {
        start++;
    }

    if (start >= height.size())
        return 0;

    int sum = 0;
    for (int i = start; i < height.size(); i++) {
        auto currentheight = height[i];
        if (buckets.empty()) {
            buckets.push(pos(i, currentheight));
        } else {
            auto top = buckets.top();
            if (currentheight <= top.height) {
                buckets.push(pos(i, currentheight));
            } else {
                int tempsum = 0;
                int currentindex = i;
                auto leftheight = 0;
                while (!buckets.empty() && buckets.top().height < currentheight) {
                    tempsum += buckets.top().height * (currentindex - buckets.top().index);
                    currentindex = buckets.top().index;
                    leftheight = buckets.top().height;
                    buckets.pop();
                }
                if (!buckets.empty()) {
                    leftheight = buckets.top().height;
                }
                buckets.push(pos(currentindex, currentheight));
                sum += std::min(leftheight, currentheight) * (i - currentindex) - tempsum;
            }
        }
    }

    return sum;
};