#include "common.h"

class Solution49 {
public:
    vector<vector<string>> groupAnagrams(vector<string> &strs) {
        int size = strs.size();

        if (size == 0)
            return vector<vector<string>>({vector<string>({""})});

        if (size == 1)
            return vector<vector<string>>({strs});

        unordered_map<string, vector<string>> map;
        for (size_t i = 0; i < size; i++) {
            auto currstr = strs[i];
            auto key = getKeyString(currstr);
            if (map.count(key) == 0) {
                map[key] = vector<string>({currstr});
            } else {
                map[key].push_back(currstr);
            }
        }

        vector<vector<string>> result;
        for (auto &&i : map)
        {
            result.push_back(i.second);
        }
        return result;
    }

    string getKeyString(string origin) {
        vector<int> count(26, 0);
        for (char c : origin) {
            count[c - 'a']++;
        }
        string key;
        for (int i = 0; i < 26; i++) {
            if (count[i] == 0)
                continue;
            key += (char)i;
            key += to_string(count[i]);
        }
        return key;
    }
};