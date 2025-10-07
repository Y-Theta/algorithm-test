#include "0-common.h"

class Solution1488 {
public:
    vector<int> avoidFlood(vector<int> &rains) {
        vector<int> floorOrder;
        unordered_map<int, int> lake;
        for (int i = 0; i < rains.size(); i++) {
            if (rains[i] == 0) {
                floorOrder.push_back(i);
            } else {
                if (lake.count(rains[i]) > 0) {
                    if (floorOrder.empty()) {
                        return vector<int>();
                    }
                    bool flag = false;
                    for (int j = 0; j < floorOrder.size(); j++) {
                        if (floorOrder[j] > lake[rains[i]]) {
                            rains[floorOrder[j]] = rains[i];
                            lake[rains[i]] = i;
                            flag = true;
                            floorOrder.erase(floorOrder.begin() + j);
                            break;
                        }
                    }
                    if (!flag) {
                        return vector<int>();
                    }
                } else {
                    lake[rains[i]] = i;
                }
                rains[i] = -1;
            }
        }

        for (int i = 0; i < rains.size(); i++) {
            if (rains[i] == 0) {
                rains[i] = 1;
            }
        }

        return rains;
    }
};