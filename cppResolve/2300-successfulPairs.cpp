#include "0-common.h"

vector<int> SolutionTest1::successfulPairs(vector<int> &spells, vector<int> &potions, long long success) {
    std::sort(potions.begin(), potions.end(), [](int a, int b) -> bool { return a < b; });

    for (int i = 0; i < spells.size(); i++) {
        int power = spells[i];
        long long need = success / power;
        if (need == 0) {
            spells[i] = potions.size();
        } else {
            if ((success % power) > 0) {
                need += 1;
            }
            if (potions[potions.size() - 1] < need) {
                spells[i] = 0;
            } else {
                int index = upper_bound(potions.begin(),potions.end(),need - 1) - potions.begin();
                spells[i] = potions.size() - index;
            }
        }
    }

    return spells;
}