#include "cppResolve/0-common.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace std;

int main() {
    // unordered_set<int> set;
    // set.insert(1);
    // set.insert(1);
    // for (int i : set) {
    //     cout << i << " ";
    // }

    SolutionTest1 test;
    vector<int> vec = {3,1,2};
    vector<int> vec1 = {8,5,8};
    test.successfulPairs(vec, vec1, 16);
}