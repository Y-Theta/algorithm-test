#include "common.h"

class LRUCache {
private:
    ListNode *first = nullptr;
    int size = 0;
    ListNode *last = nullptr;
    unordered_map<int, ListNode *> itemmap;
    int capacity;

public:
    LRUCache(int capacity) {
        this->capacity = capacity;
    }

    int get(int key) {
        if (itemmap.count(key) <= 0)
            return -1;
        auto node = itemmap[key];
        if (node == first)
            return node->val;
        if (node->next != nullptr) {
            node->next->pre = node->pre;
        }
        if (node->pre != nullptr) {
            node->pre->next = node->next;
        }
        if (node->next == nullptr) {
            last = node->pre;
        }
        if (node->pre != nullptr) {
            node->next = first;
            node->pre = nullptr;
            first->pre = node;
            first = node;
        }
        return node->val;
    }

    void put(int key, int value) {
        if (itemmap.count(key) > 0) {
            get(key);
            itemmap[key]->val = value;
        } else {
            ListNode *node;
            if (size == 0) {
                last = new ListNode();
                last->val = value;
                last->key = key;
                first = last;
                node = last;
                size++;
            } else if (size < this->capacity) {
                node = new ListNode();
                node->val = value;
                node->key = key;
                node->next = first;
                first->pre = node;
                first = node;
                size++;
            } else {
                node = new ListNode();
                node->val = value;
                node->key = key;
                itemmap.erase(last->key);
                if (last == first) {
                    last = first = node;
                } else {
                    last = last->pre;
                    last->next = nullptr;
                    node->next = first;
                    first->pre = node;
                    first = node;
                }
            }
            itemmap[key] = node;
        }
    }
};
