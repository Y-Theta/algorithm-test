from Common import ListNode
from typing import Optional
from typing import List

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

    def hasSameDigits(self, s: str) -> bool:
        dataarr = list(map(lambda c: ord(c) - ord("0"), s))
        count = len(dataarr)
        while count > 2:
            for i in range(1, count):
                dataarr[i - 1] = (dataarr[i] + dataarr[i - 1]) % 10
            count -= 1
        return dataarr[0] == dataarr[1]
    
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        numdict = dict()
        for i in range(len(nums)):
            num = nums[i]
            if num in numdict:
                if i - numdict[num] <= k :
                    return True
            numdict[num] = i
        return False
    