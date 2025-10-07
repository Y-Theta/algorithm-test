#include "0-common.h"

int SolutionTest1::longestCommonSubsequence(string text1, string text2) {
    int dp[text1.size()][text2.size()];
    // dp[i][j] = text1[i] == text2[j] ? dp[i-1][j-1] + 1 : max{dp[i][j-1] , dp[i-1][j]}

    for (int i = 0; i < text1.size(); i++) {
        for (int j = 0; j < text2.size(); j++) {
            if (i == 0 && j == 0) {
                dp[i][j] = text1[0] == text2[0] ? 1 : 0;
            } else if (i == 0) {
                dp[0][j] = text1[0] == text2[j] ? 1 : dp[i][j - 1];
            } else if (j == 0) {
                dp[i][0] = text1[i] == text2[0] ? 1 : dp[i - 1][j];
            } else {
                dp[i][j] = text1[i] == text2[j] ? dp[i - 1][j - 1] + 1 : std::max(dp[i - 1][j], dp[i][j - 1]);
            }
        }
    }

    return dp[text1.size() - 1][text2.size() - 1];
}