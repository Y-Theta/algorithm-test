from Common import ListNode, TreeNode
from typing import Optional, List, Dict, Counter
from math import gcd, sqrt, inf


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

    # region Solution 2048
    def getDigit(self, n: int) -> int:
        if n == 0:
            return 0
        return 1 + self.getDigit(n // 10)

    def isSuit(
        self,
        aim: int,
        picked: int,
        total: int,
        result: int,
        nums: List[int],
        used: List[bool],
    ) -> bool:
        # 检查是否所有数字都已被选择
        if picked == total:
            return result
        # 遍历所有可能的选择
        for i in range(total):
            # 如果当前数字已使用，或者当前数字与前一个数字相同并且前一个数字未被使用，则跳过
            if used[i] or (i > 0 and nums[i] == nums[i - 1] and not used[i - 1]):
                continue
            # 选择当前数字，并标记为已使用
            used[i] = True
            result = result * 10 + nums[i]
            picked += 1  # selNums 增加
            # 递归调用，继续选择下一个数字
            tempresult = self.isSuit(aim, picked, total, result, nums, used)
            if tempresult > aim:
                return tempresult
            # 回溯：撤销选择，并将数字标记为未使用
            result = (result - nums[i]) // 10
            picked -= 1  # selNums 减少
            used[i] = False
        return -1

    def getBeautifulNumberElement(
        self, digit: int, sum: int, tempresult: List[int], result: List[List[int]]
    ):
        if sum == digit:
            toappend = True
            for i in result:
                if sorted(i) == sorted(tempresult):
                    toappend = False
                    break
            if toappend:
                result.append(tempresult)
        for i in range(1, digit + 1):
            temparr = []
            temparr += tempresult
            if i in temparr:
                continue
            if i + sum > digit:
                break
            temparr.append(i)
            self.getBeautifulNumberElement(digit, sum + i, temparr, result)

    def nextBeautifulNumber(self, n: int) -> int:
        oridigit = self.getDigit(n)
        digit = oridigit
        results: List[int] = []
        while digit <= oridigit + 1:
            tempresult = []
            self.getBeautifulNumberElement(digit, 0, [], tempresult)
            for i in tempresult:
                temparr = []
                for k in i:
                    for u in range(k):
                        temparr.append(k)
                result = self.isSuit(
                    n, 0, len(temparr), 0, temparr, [False] * len(temparr)
                )
                if result > n:
                    results.append(result)
                    continue
            digit += 1
        return min(results)

    # endregion

    # region Solution 234
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        stack = []
        slow = head
        fast = head
        middle = False
        while slow is not None:
            if not middle:
                stack.append(slow.val)
            else:
                if slow.val != stack.pop():
                    return False
            slow = slow.next
            if not middle:
                if fast.next is None:
                    middle = True
                    stack.pop()
                    continue
                elif fast.next.next is None:
                    middle = True
                    continue
                fast = fast.next.next
        return True

    # endregion

    # region Solution 242
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        dict1 = [0] * 26
        for c in s:
            dict1[ord(c) - ord("a")] += 1

        for c in t:
            dict1[ord(c) - ord("a")] -= 1

        for k in dict1:
            if k != 0:
                return False

        return True

    # endregion

    # region Solution 243
    def shortestDistance(self, wordsDict: List[str], word1: str, word2: str) -> int:
        start1 = -1
        start2 = -1
        minspan = len(wordsDict)
        for index in range(len(wordsDict)):
            word = wordsDict[index]
            if word == word1:
                start1 = index
                if start2 >= 0:
                    minspan = min(minspan, start1 - start2)
            elif word == word2:
                start2 = index
                if start1 >= 0:
                    minspan = min(minspan, start2 - start1)

        return minspan

    # endregion

    # region Solution 246
    def isStrobogrammatic(self, num: str) -> bool:
        for index in range((len(num) // 2) + 1):
            match num[index]:
                case "0":
                    if num[len(num) - index - 1] != "0":
                        return False
                case "1":
                    if num[len(num) - index - 1] != "1":
                        return False
                case "6":
                    if num[len(num) - index - 1] != "9":
                        return False
                case "9":
                    if num[len(num) - index - 1] != "6":
                        return False
                case "8":
                    if num[len(num) - index - 1] != "8":
                        return False
                case _:
                    return False
        return True

    # endregion

    # region Solution 252
    def canAttendMeetings(self, intervals: List[List[int]]) -> bool:
        if len(intervals) < 2:
            return True
        intervals.sort(key=lambda x: x[0])
        for index in range(1, len(intervals)):
            if intervals[index][0] < intervals[index - 1][1]:
                return False
        return True

    # endregion

    # region Solution 257
    def leftTraversal(self, root: Optional[TreeNode], stack: str, path: List[str]):
        if root is None:
            return
        stack = f"{stack}->{root.val}"
        self.leftTraversal(root.left, stack, path)
        self.leftTraversal(root.right, stack, path)
        if root.left == None and root.right == None:
            path.append(stack[2:])

    def binaryTreePaths(self, root: Optional[TreeNode]) -> List[str]:
        result = []
        self.leftTraversal(root, "", result)
        return result

    # endregion

    # region Solution 433
    def offsetG(self, str1: str, str2: str) -> int:
        k = 0
        for i in range(len(str1)):
            if str1[i] != str2[i]:
                k += 1
        return k

    def minMutation(self, startGene: str, endGene: str, bank: List[str]) -> int:
        aim = self.offsetG(startGene, endGene)
        if aim <= 1:
            return aim
        if bank is None or len(bank) == 0:
            return -1
        mdict: Dict[int, set[str]] = dict()
        for g in bank:
            k = self.offsetG(endGene, g)
            if k not in mdict:
                mdict[k] = set()
            mdict[k].add(g)

        maxkey = max(mdict.keys())
        if aim > maxkey + 1:
            return -1

        remain = set()
        for i in range(len(bank)):
            pre = None
            if i - 1 > 0 and i - 1 in mdict:
                pre = mdict[i - 1]
            if i in mdict:
                layer = mdict[i]
            else:
                layer = list(remain)
                remain.clear()
            if pre is not None:
                mdict[i] = set()
                for item in pre:
                    for g in layer:
                        if self.offsetG(g, item) == 1:
                            mdict[i].add(g)
                        else:
                            remain.add(g)
            if i not in mdict:
                return -1
            if i + 1 >= aim:
                for item in mdict[i]:
                    if self.offsetG(item, startGene) <= 1:
                        return i + 1
            if len(mdict[i]) == 0 and i + 1 < aim:
                return -1

        return -1

    # endregion

    # region Solution 383
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        dict2 = [0] * 26
        for i in magazine:
            dict2[ord(i) - ord("a")] += 1

        for i in ransomNote:
            dict2[ord(i) - ord("a")] -= 1
            if dict2[ord(i) - ord("a")] < 0:
                return False
        return True

    # endregion

    # region Solution 1716
    def totalMoney(self, n: int) -> int:
        turn = n // 7
        base = ((4 + turn + 3) * turn // 2) * 7
        remain = n % 7
        other = ((turn + 1) + (turn + remain)) * remain // 2
        return base + other

    # endregion

    # region Solution 2464

    def validSubarraySplit(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [1001] * (n + 1)
        dp[0] = 0
        for i in range(1, n + 1):
            for j in range(1, i + 1):
                if gcd(nums[i - 1], nums[j - 1]) > 1:
                    dp[i] = min(dp[i], dp[j - 1] + 1)
        return -1 if dp[n] == 1001 else dp[n]

    # endregion

    # region Solution 2125
    def numberOfBeams(self, bank: List[str]) -> int:
        preline = 0
        sum = 0
        for line in bank:
            currentline = line.count("1")
            sum += preline * currentline
            if currentline > 0:
                preline = currentline
        return sum

    # endregion

    # region Solution 22
    def generateParenthesis(self, n: int) -> List[str]:
        return

    # endregion

    # region Solution 45
    def jump(self, nums: List[int]) -> int:
        # dp[i] = min ( if nums[i-k] >= i-k dp[i-k] + 1 )
        dp = [10000] * len(nums)
        dp[0] = 0

        for i in range(0, len(nums)):
            num = nums[i]
            maxrange = min(i + num + 1, len(nums))
            maxrangeleft = min(i + 1, len(nums))
            if maxrange >= len(nums):
                return dp[i] + 1
            for k in range(maxrangeleft, maxrange):
                dp[k] = min(dp[k], dp[i] + 1)
                if k == len(nums) - 1:
                    return dp[k]
            # for j in range(0, i):
            #     if nums[j] >= i - j:
            #         dp[i] = min(dp[i], dp[j] + 1)

        return dp[i - 1]

    # endregion

    # region Solution 63
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        dp = [
            [0 for _ in range(len(obstacleGrid[0]))] for _ in range(len(obstacleGrid))
        ]
        dp[0][0] = 1 if obstacleGrid[0][0] == 0 else 0
        for r in range(1, len(obstacleGrid)):
            dp[r][0] = dp[r - 1][0] if obstacleGrid[r][0] == 0 else 0

        for c in range(1, len(obstacleGrid[0])):
            dp[0][c] = dp[0][c - 1] if obstacleGrid[0][c] == 0 else 0

        for r in range(1, len(obstacleGrid)):
            for c in range(1, len(obstacleGrid[0])):
                dp[r][c] = (
                    (dp[r - 1][c] + dp[r][c - 1]) if obstacleGrid[r][c] == 0 else 0
                )

        return dp[len(obstacleGrid) - 1][len(obstacleGrid[0]) - 1]

    # endregion

    # region Solution 72
    def minDistance(self, word1: str, word2: str) -> int:
        dp = [[0 for _ in range(len(word1) + 1)] for _ in range(len(word2) + 1)]
        dp[0][0] = 0

        # 从 word 变成 0 要删除的字符数
        for i in range(1, len(word1) + 1):
            dp[0][i] = i
        for i in range(1, len(word2) + 1):
            dp[i][0] = i

        for w1 in range(1, len(word1) + 1):
            for w2 in range(1, len(word2) + 1):
                dp[w2][w1] = (
                    dp[w2 - 1][w1 - 1]
                    if word1[w1 - 1] == word2[w2 - 1]
                    else min(dp[w2][w1 - 1], dp[w2 - 1][w1 - 1], dp[w2 - 1][w1]) + 1
                )

        return dp[len(word2)][len(word1)]

    # endregion

    # region Solution 91
    def numDecodings(self, s: str) -> int:
        dp = [0] * (len(s) + 1)
        dp[0] = 1
        for i in range(1, len(s) + 1):
            if s[i - 1] != "0":
                dp[i] = dp[i - 1]
            if i - 2 >= 0 and s[i - 2] != "0" :
                num = int(s[i - 2:i])
                if num >= 1 and num <= 26:
                    dp[i] += dp[i - 2]

        return dp[len(s)]

    # endregion
    
    # region Solution 97
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        
        return
    # endregion
    
    # region Solution 3354
    def countValidSelections(self, nums: List[int]) -> int:
        total = sum(nums)
        temptotal = 0
        count = 0
        for i in range(len(nums)):
            temptotal += nums[i]
            if nums[i] == 0:
                left = total - temptotal
                if temptotal == left:
                    count += 2
                elif abs(temptotal - left) == 1:
                    count += 1
        return count
    # endregion