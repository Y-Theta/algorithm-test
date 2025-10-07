#include "0-common.h"

class Solution62 {
public:
    int uniquePaths(int m, int n) {
        int dp[m][n];

        for (int r = 0; r < m; r++) {
            for (int c = 0; c < n; c++) {
                if (r == 0 && c == 0) {
                    dp[0][0] = 1;
                } else if (r == 0) {
                    dp[r][c] = dp[r][c - 1];
                } else if (c == 0) {
                    dp[r][c] = dp[r - 1][c];
                } else {
                    dp[r][c] = dp[r - 1][c] + dp[r][c - 1];
                }
            }
        }

        return dp[m - 1][n - 1];
    }
};