from Common import ListNode
from typing import Optional


class Solution1:

    def hello():
        print("wwww")
        return

    def removeElements_203(
        self, head: Optional[ListNode], val: int
    ) -> Optional[ListNode]:
        if head is None:
            return head

        first = head
        pre: Optional[ListNode] = None
        while head != None:
            if head.val == val:
                if head.next != None:
                    head.val = head.next.val
                    head.next = head.next.next
                else:
                    if pre == None:
                        first = None
                    else:
                        pre.next = None
                    head = None
            else:
                pre = head
                head = head.next

        return first

    def isIsomorphic(self, s: str, t: str) -> bool:
        sset: dict = dict()
        sexist: set = set()
        news = ""
        for i in range(len(s)):
            if not s[i] in sset:
                sset[s[i]] = t[i]
                if sset[s[i]] in sexist:
                    return False
                sexist.add(sset[s[i]])
            news += sset[s[i]]
        return news == t

    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        pre = None
        while head is not None:
            oldhead = head
            head = head.next
            oldhead.next = pre
            if head is None:
                return oldhead
            pre = oldhead
        return head
