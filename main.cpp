#include "cppResolve/common.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace std;

struct test {
    int a = 1;
};

const test* testfunc(test* x) {
    return new test();
}

int main() {
    // unordered_set<int> set;
    // set.insert(1);
    // set.insert(1);
    // for (int i : set) {
    //     cout << i << " ";
    // }
    test t;
    test *p = &t;
    auto x = testfunc(p);
    x = new test();

    Solution48 sln;
    vector<vector<int>> vector = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    sln.rotate(vector);
}