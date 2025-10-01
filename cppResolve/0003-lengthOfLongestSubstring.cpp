#include "0-common.h"

int SolutionTest1::lengthOfLongestSubstring(string s) {
    int windowStart = -1;
    int windowEnd = 0;
    int max = 1;
    int charset[256];
    int size = sizeof(charset) / sizeof(int);
    std::fill(charset, charset + 256, -1);
    charset[s[0]] = -1;

    auto stringLength = s.length();
    if (stringLength == 0) {
        return 0;
    }
    for (int i = 0; i < stringLength; i++) {
        auto currentChar = s[i];
        windowEnd = i;
        if (charset[currentChar] >= windowStart) {
            max = std::max(max, windowEnd - windowStart);
            windowStart = charset[currentChar] + 1;
        }
        charset[currentChar] = i;
    }
    max = std::max(max, windowEnd + 1 - windowStart);

    return max;
}
