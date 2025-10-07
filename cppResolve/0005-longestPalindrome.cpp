#include "0-common.h"

int isPalindrome(string str, int index, int max, vector<vector<int>> word) {
    if (index - max >= 0) {
        auto c = str[index];
        if (word[c].size() > 0) {
            for (int i = 0; i < word[c].size(); i++) {
                bool flag = false;
                if (index - word[c][i] >= max) {
                    int tempstart = word[c][i];
                    int tempend = index;
                    flag = true;
                    while (tempstart <= tempend) {
                        if (str[tempstart] != str[tempend]) {
                            flag = false;
                            break;
                        }
                        tempstart++;
                        tempend--;
                    }
                }
                if (flag) {
                    return word[c][i];
                }
            }
        }
    }
    return -1;
}

string SolutionTest1::longestPalindrome(string str) {
    // dp[i] = max(dp[i-1], end with i)
    vector<vector<int>> word(128, vector<int>());

    int max = 1;
    int maxstart = 0;

    for (int i = 0; i < str.size(); i++) {
        auto c = str[i];
        word[c].push_back(i);

        auto start = isPalindrome(str, i, max, word);
        if (start >= 0 && (i - start + 1) > max) {
            max = i - start + 1;
            maxstart = start;
        }
    }

    return str.substr(maxstart, max);
}