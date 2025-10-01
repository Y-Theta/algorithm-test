#include "0-common.h"

bool SolutionTest1::wordBreak(string s, vector<string> &wordDict) {
    vector<int> dp(s.length() + 1, 0);
    // dp[i] = 1 only if dp[i - s.length] == 1 && s.substr(i - s.length, s.length) in wordDict
    dp[0] = 1;
    string *worddicsorted = wordDict.data();
    sort(worddicsorted, worddicsorted + wordDict.size(), [](string a, string b) -> bool { return a.size() < b.size(); });
    for (int i = wordDict[0].size(); i <= s.length(); i++) {
        
        for (int j = 0; j < wordDict.size(); j++)
        {
            auto word = wordDict[j];
            auto lastpos = i - (int)word.size();
            if (lastpos < 0)
                continue;
            auto subs = s.substr(lastpos, (int)word.size());
            if (dp[lastpos] > 0 && subs == word) {
                dp[i] = 1;
                break;
            }
        }
    }

    return dp[s.length()] > 0;
};