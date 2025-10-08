#include "0-common.h"

class Solution14 {
public:
    string longestCommonPrefix(vector<string> &strs) {
        string pattern = strs[0];

        for (int i = 1; i < strs.size(); i++) {
            if (strs[i].size() == 0)
                return "";

            if (pattern.size() == 0)
                return "";

            for (int j = 0; j < pattern.size(); j++) {
                if (pattern[j] != strs[i][j]) {
                    pattern = pattern.substr(0, j);
                    break;
                }
            }
        }

        return pattern;
    }
};