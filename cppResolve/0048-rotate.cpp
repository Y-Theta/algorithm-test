#include "common.h"

struct pt {
    int x;
    int y;

    pt(int x, int y) : x(x), y(y) {};
};


pt rotateitem(int x, int y, int offset, int half) {

    // y=x
    auto nx = y;
    auto ny = x;

    // y
    nx = 2 * half - nx + offset;

    return pt(nx, ny);
}

void Solution48::rotate(vector<vector<int>> &matrix) {
    if (matrix.size() == 1)
        return;

    int n = matrix.size();
    int offset = -1 + n % 2;
    int half = n / 2;

    for (int y = 0; y < half; y++) {
        for (int x = y; x < matrix.size() - y - 1; x++) {
            auto npt = pt(x, y);
            auto val = matrix[y][x];
            while (true) {
                npt = rotateitem(npt.x, npt.y, offset, half);
                auto c = matrix[npt.y][npt.x];
                matrix[npt.y][npt.x] = val;
                val = c;
                if (npt.x == x && npt.y == y) {
                    break;
                }
            }
        }
    }
};
