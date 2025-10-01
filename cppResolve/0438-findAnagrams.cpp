#include "0-common.h"

class Solution438 {
public:
    vector<int> findAnagrams(string s, string p) {
        if (s.size() == 0)
            return vector<int>({0});
        if (p.size() > s.size())
            return vector<int>();
        vector<int> result;
        auto offset = p.size() - 1;
        auto aimkey = getKeyString(p, 0, offset);
        auto currentkey = getKeyString(s, 0, offset);
        if (comparerKey(aimkey, currentkey)) {
            result.push_back(0);
        }
        for (size_t i = 1; i <= s.size() - p.size(); i++) {
            currentkey[s[i - 1] - 'a']--;
            currentkey[s[i + offset] - 'a']++;
            if (comparerKey(aimkey, currentkey)) {
                result.push_back(i);
            }
        }
        return result;
    }

    vector<int> getKeyString(string origin, int start, int end) {
        vector<int> count(26, 0);
        for (int i = start; i <= end; i++) {
            count[origin[i] - 'a']++;
        }
        return count;
    }

    bool comparerKey(vector<int> a, vector<int> b) {
        return a == b;
    }
};