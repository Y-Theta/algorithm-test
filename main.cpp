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
    vector<int> vec = {1,7,5};
    cout << test.trap(vec) << endl;
}