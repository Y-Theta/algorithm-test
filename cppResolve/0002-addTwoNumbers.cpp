#include "common.h"

class Solution2 {
public:
    ListNode *addTwoNumbers(ListNode *l1, ListNode *l2) {
        ListNode *l1cur = l1;
        ListNode *l2cur = l2;
        ListNode *l1tail = nullptr;

        bool addone = false;
        for (int i = 0;; i++) {
            auto sum = 0;
            if (l1cur != nullptr) {
                sum += l1cur->val;
            }
            if (l2cur != nullptr) {
                sum += l2cur->val;
            }
            if (addone) {
                sum += 1;
            }
            auto newval = sum % 10;
            addone = (sum / 10) > 0;
            if (l2cur != nullptr) {
                if (l1tail != nullptr) {
                    l2cur->val = newval;
                    l1tail->next = l2cur;
                    l1tail = l2cur;
                }
                l2cur = l2cur->next;
            }
            if (l1cur != nullptr) {
                l1cur->val = newval;
                if (l1cur->next == nullptr) {
                    l1tail = l1cur;
                }
                l1cur = l1cur->next;
            }
            if (l1cur == nullptr && l2cur == nullptr) {
                if (addone) {
                    l1tail->next = new ListNode();
                    l1tail->next->val = 1;
                }
                break;
            }
        }

        return l1;
    }
};