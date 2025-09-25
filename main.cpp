#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include "cppResolve/common.h"

using namespace std;

int main() {
    Solution128 sln3;
    auto val = vector<int>{100, 4, 200, 1, 3, 2};
    sln3.longestConsecutive(val);
}