#include <algorithm>
#include <bitset>
#include <iostream>
#include <stack>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct ListNode {
    int val;
    ListNode *next = nullptr;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

struct ListNode *addTwoInteger(struct ListNode *a, struct ListNode *b) {
    ListNode *nlist = nullptr;
    ListNode *ncurrent = nullptr;
    int sum = 0;
    int over10 = 0;
    while (a != nullptr || b != nullptr) {
        sum = over10;
        if (a != nullptr) {
            sum += a->val;
        }
        if (b != nullptr) {
            sum += b->val;
        }
        over10 = sum / 10;
        sum = sum % 10;

        auto c = new ListNode();
        c->val = sum;
        if (nlist == nullptr) {
            nlist = c;
            ncurrent = c;
        } else {
            ncurrent->next = c;
            ncurrent = c;
        }
        if (a != nullptr) {
            a = a->next;
        }
        if (b != nullptr) {
            b = b->next;
        }
    }

    if (over10) {
        ncurrent->next = new ListNode();
        ncurrent->next->val = 1;
        ncurrent->next->next = nullptr;
    }

    return nlist;
}