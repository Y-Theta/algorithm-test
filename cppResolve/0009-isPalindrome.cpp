#include "0-common.h"

bool SolutionTest1::isPalindrome(int x) {
    if (x < 0)
        return false;

    int digit = 0;
    int temp = x;
    while (temp != 0) {
        temp = temp / 10;
        digit++;
    }

    int left = 0;
    int right = digit - 1;

    while (right >= left) {
        if (((x / (int)std::pow(10, right)) % 10) != ((x / (int)std::pow(10, left)) % 10)) {
            return false;
        }
        right--;
        left++;
    }

    return true;
};