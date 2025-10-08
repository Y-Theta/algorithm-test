#include "0-common.h"

int quickfind(vector<int> array, int aim, int start, int end) {
    if (start == end){
        return start;
    }
    int mid = (start + end) / 2;
    if (array[mid] >= aim) {
        return quickfind(array, aim, start, mid);
    } else {
        return quickfind(array, aim, mid, end);
    }
}

vector<int> SolutionTest1::successfulPairs(vector<int> &spells, vector<int> &potions, long long success) {
    auto potionsdata = potions.data();
    std::sort(potionsdata, potionsdata + potions.size(), [](int a, int b) -> bool { return a < b; });

    for (int i = 0; i < spells.size(); i++) {
        int power = spells[i];
        int need = success / power;
        if (need == 0) {
            spells[i] = potions.size();
        } else {
            if (success % power) {
                need += 1;
            }
            if (potions[potions.size() - 1] < need) {
                spells[i] = 0;
            } else {
                int index = quickfind(potions, need, 0, potions.size() - 1);
                spells[i] = potions.size() - index;
            }
        }
    }

    return spells;
}