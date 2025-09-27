#include "common.h"

void Solution73::setZeroes(vector<vector<int>> &matrix) {
    int m = matrix.size();
    int n = matrix[0].size();

    unordered_set<int> rset;
    unordered_set<int> cset;
    for (int r = 0; r < m; r++) {
        for (int c = 0; c < n; c++) {
            if (matrix[r][c] == 0) {
                rset.insert(r);
                cset.insert(c);
            }
        }
    }

    for (int r : rset) {
        for (int c = 0; c < n; c++) {
            matrix[r][c] = 0;
        }
    }

    for (int c : cset) {
        for (int r = 0; r < m; r++) {
            matrix[r][c] = 0;
        }
    }
};