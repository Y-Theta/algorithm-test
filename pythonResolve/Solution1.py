from Common import ListNode, TreeNode
from typing import Optional, List


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
                if i - numdict[num] <= k:
                    return True
            numdict[num] = i
        return False

    # region
    def countTranversal(self, root: Optional[TreeNode]) -> int:
        if root == None:
            return 0
        return 1 + self.countTranversal(root.left) + self.countTranversal(root.right)

    def countNodes(self, root: Optional[TreeNode]) -> int:
        return self.countTranversal(root)

    # endregion

    # region Solution 226
    def invertTreeNode(self, root: Optional[TreeNode]):
        if root is None:
            return
        orileft = root.left
        root.left = root.right
        self.invertTreeNode(root.left)
        root.right = orileft
        self.invertTreeNode(root.right)

    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        self.invertTreeNode(root)
        return root

    # endregion

    # region Solution 228
    def summaryRanges(self, nums: List[int]) -> List[str]:
        result: List[str] = []
        start = 0
        for i in range(1, len(nums)):
            if nums[i] - nums[i - 1] > 1:
                if nums[start] == nums[i - 1]:
                    result.append(f"{nums[start]}")
                else:
                    result.append(f"{nums[start]}->{nums[i-1]}")
                start = i
        if start < len(nums):
            if nums[start] == nums[i - 1]:
                result.append(f"{nums[start]}")
            else:
                result.append(f"{nums[start]}->{nums[len(nums) - 1]}")
        return result

    # endregion

    # region Solution 231
    def isPowerOfTwo(self, n: int) -> bool:
        return n > 0 and (n - 1) & n == 0

    # endregion

    # region Solution 159
    def lengthOfLongestSubstringTwoDistinct(self, s: str) -> int:
        start = 0
        maxrange = 0
        charlist: List[int] = [-1] * 128
        top = None
        count = 0
        torem = None
        for i in range(len(s)):
            if charlist[ord(s[i])] > -1:
                if s[i] is not top:
                    charlist[ord(s[i])] = i
                    torem = top
            else:
                if count >= 2:
                    maxrange = max(maxrange, i - start)
                    charlist[ord(torem)] = -1
                    start = charlist[ord(s[i - 1])]
                else:
                    count += 1
                charlist[ord(s[i])] = i
                torem = s[i - 1]
            top = s[i]

        maxrange = max(maxrange, len(s) - start)
        return maxrange

    # endregion

    # region Solution 349
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        set1 = set(nums1)
        return list(set1.intersection(nums2))

    # endregion

    # region Solution 350
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        dict1 = dict()
        for num in nums1:
            if num not in dict1:
                dict1[num] = 0
            dict1[num] += 1

        dict2 = dict()
        for num in nums2:
            if num not in dict2:
                dict2[num] = 0
            dict2[num] += 1

        result = []
        intersect = set(dict1.keys()).intersection(dict2.keys())
        for i in intersect:
            count = min(dict1[i], dict2[i])
            for k in range(count):
                result.append(i)
        return result

    # endregion
