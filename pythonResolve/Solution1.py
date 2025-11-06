from Common import ListNode, TreeNode, SegmentTreeNode, Pos
from typing import Optional, List, Dict, Counter, Tuple
from math import gcd, sqrt, inf, factorial
from dataclasses import dataclass


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
            if i - 2 >= 0 and s[i - 2] != "0":
                num = int(s[i - 2 : i])
                if num >= 1 and num <= 26:
                    dp[i] += dp[i - 2]

        return dp[len(s)]

    # endregion

    # region Solution 97
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        if len(s1) + len(s2) != len(s3):
            return False
        # dp[i][j] =  s1[i] = s3[i+j] dp[i-1][j]  s2[j] = s3[i+j] dp[i][j-1]
        dp = [[0 for _ in range(len(s1) + 1)] for _ in range(len(s2) + 1)]
        dp[0][0] = 1
        for i in range(1, len(s2) + 1):
            dp[i][0] = dp[i - 1][0] if s2[i - 1] == s3[i - 1] else 0
        for i in range(1, len(s1) + 1):
            dp[0][i] = dp[0][i - 1] if s1[i - 1] == s3[i - 1] else 0

        for i in range(1, len(s1) + 1):
            for j in range(1, len(s2) + 1):
                if s1[i - 1] == s3[i + j - 1] and dp[j][i - 1] == 1:
                    dp[j][i] = dp[j][i - 1]
                if s2[j - 1] == s3[i + j - 1] and dp[j - 1][i] == 1:
                    dp[j][i] = dp[j - 1][i]

        return dp[len(s2)][len(s1)] > 0

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

    # region Solution 120
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        layer = triangle[len(triangle) - 1]
        for i in range(len(triangle) - 2, -1, -1):
            currentlayer = triangle[i]
            currentlength = len(currentlayer)
            for j in range(currentlength):
                layer[j] = min(
                    layer[j] + currentlayer[j], layer[j + 1] + currentlayer[j]
                )
            layer[j + 1] = inf
        return min(layer)

    # endregion

    # region Solution 122
    def maxProfit(self, prices: List[int]) -> int:
        ls = list()
        ls.append(prices[0])
        profit = 0
        for i in range(1, len(prices)):
            num = prices[i]
            if num >= ls[-1]:
                ls.append(num)
            else:
                profit += ls[-1] - ls[0]
                ls.clear()
                ls.append(num)

        if len(ls) > 0:
            profit += ls[-1] - ls[0]

        return profit

    # endregion

    # region Solution 131
    def is_pri(self, s: str, start: int, end: int) -> bool:
        if start <= end:
            return s[start] == s[end] and self.is_pri(s, start + 1, end - 1)
        else:
            return True

    def dfs_131(self, s: str, tempresult: List[str], result: List[List[str]]):
        for i in range(len(s)):
            subs = s[0 : i + 1]
            if self.is_pri(subs, 0, len(subs) - 1):
                tempresult.append(subs)
                if i < len(s) - 1:
                    self.dfs_131(s[i + 1 : len(s) + 1], list(tempresult), result)
                else:
                    result.append(tempresult)
                tempresult.pop()

    def partition(self, s: str) -> List[List[str]]:
        result = []
        self.dfs_131(s, [], result)
        return result

    # endregion

    # region Solution 213
    class struct_213:
        def __init__(self, sum: int, sumstart: int):
            self.sum = sum
            self.sumstart = sumstart

    def rob(self, nums: List[int]) -> int:
        dp: List["Solution1.struct_213"] = [
            Solution1.struct_213(0, 0) for _ in range(len(nums))
        ]
        dp[0].sum = 0
        dp[0].sumstart = nums[0]
        for i in range(1, len(nums)):
            dp[i].sum = max(dp[i - 1].sum, nums[i])
            dp[i].sumstart = max(dp[i - 1].sumstart, nums[i])
            if i - 2 >= 0:
                dp[i].sum = max(dp[i].sum, dp[i - 2].sum + nums[i])
                if i == len(nums) - 1:
                    dp[i].sumstart = max(dp[i].sumstart, dp[i - 2].sumstart)
                else:
                    dp[i].sumstart = max(dp[i].sumstart, dp[i - 2].sumstart + nums[i])

        return max(dp[len(nums) - 1].sum, dp[len(nums) - 1].sumstart)

    # endregion

    # region Solution 338
    def countBits(self, n: int) -> List[int]:
        dp = [0] * (n + 1)
        dp[0] = 0
        for i in range(1, n + 1):
            dp[i] = dp[i >> 1] + (i & 1)
        return dp

    # endregion

    # region Solution 3370
    def smallestNumber(self, n: int) -> int:
        num = n
        digit = 0
        while num > 0:
            digit += 1
            num = num >> 1

        return (2 ** (digit)) - 1

    # endregion

    # region Solution 187

    def findRepeatedDnaSequences(self, s: str) -> List[str]:
        length = len(s)
        if length <= 10:
            return []
        result = set()
        keyset = set()
        for i in range(11, length + 1):
            key = s[i - 10, i]
            if key in keyset:
                result.add(key)
            else:
                keyset.add(key)

        return list(result)

    # endregion

    # region Solution 718
    def findLength(self, nums1: List[int], nums2: List[int]) -> int:
        dp = [[0 for _ in range(len(nums1))] for _ in range(len(nums2))]
        dp[0][0] = 1 if nums1[0] == nums2[0] else 0
        resultmax = dp[0][0]
        for i in range(1, len(nums1)):
            dp[0][i] = 1 if nums1[i] == nums2[0] else 0
            resultmax = max(resultmax, dp[0][i])
        for i in range(1, len(nums2)):
            dp[i][0] = 1 if nums2[i] == nums1[0] else 0
            resultmax = max(resultmax, dp[i][0])

        for i in range(1, len(nums1)):
            for j in range(1, len(nums2)):
                if nums1[i] == nums2[j]:
                    dp[j][i] = dp[j - 1][i - 1] + 1
                    resultmax = max(resultmax, dp[j][i])
                # else:
                #     prematch = dp[j-1][i]
                #     flag1 = True
                #     for k in range(j, j - prematch - 1,-1):
                #         if nums2[k] != nums2[k - 1]:
                #             flag1 = False
                #             break
                #     if flag1:
                #         dp[j][i] = dp[j-1][i]
                #         continue

                #     prematch = dp[j][i - 1]
                #     flag1 = True
                #     for k in range(i, i - prematch - 1, -1):
                #         if nums1[k] != nums1[k - 1]:
                #             flag1 = False
                #             break
                #     if flag1:
                #         dp[j][i] = dp[j][i - 1]
                #         continue

        return resultmax

    # endregion

    # region Solution 238
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        pretimes = [0] * len(nums)
        posttimes = [0] * len(nums)
        result = [0] * len(nums)

        pretimes[0] = 1
        posttimes[len(nums) - 1] = 1
        for i in range(1, len(nums)):
            pretimes[i] = pretimes[i - 1] * nums[i - 1]

        result[len(nums) - 1] = pretimes[len(nums) - 1]
        for i in range(len(nums) - 2, -1, -1):
            posttimes[i] = posttimes[i + 1] * nums[i + 1]
            result[i] = pretimes[i] * posttimes[i]

        return result

    # endregion

    # region Solution 258
    def addDigits(self, num: int) -> int:
        if num == 0:
            return 0
        return (num - 1) % 9 + 1

    # endregion

    # region Solution 263
    def isUgly(self, n: int) -> bool:
        if n <= 0:
            return False
        if n == 1 or n == -1:
            return True
        for i in range(1, int(sqrt(abs(n))) + 1):
            if n % i == 0:
                left = n // i
                if left % 2 != 0 and left % 3 != 0 and left % 5 != 0:
                    return False
                if i > 1 and i % 2 != 0 and i % 3 != 0 and i % 5 != 0:
                    return False
        return True

    # endregion

    # region Solution 266
    def canPermutePalindrome(self, s: str) -> bool:
        chardict = [0] * 26

        for i in range(len(s)):
            chardict[ord(s[i]) - ord("a")] += 1

        flag = True
        for kv in chardict:
            if kv % 2 != 0:
                if not flag:
                    return False
                flag = False

        return True

    # endregion

    # region Solution 278
    def isBadVersion(version: int) -> bool:
        return

    def firstBadVersionSearch(self, start: int, end: int) -> int:
        while start < end:
            mid = (start + end) // 2
            flag = self.isBadVersion(mid)
            if not flag:
                start = mid + 1
            else:
                end = mid

        return start

    def firstBadVersion(self, n: int) -> int:
        return self.firstBadVersionSearch(0, n)

    # endregion

    # region Solution 290
    def wordPattern(self, pattern: str, s: str) -> bool:
        seles = s.split(" ")
        plength = len(pattern)
        slength = len(seles)
        if plength != slength:
            return False

        pdict = dict()
        up = set()
        for i in range(plength):
            if pattern[i] in pdict:
                if seles[i] != pdict[pattern[i]]:
                    return False
            else:
                if seles[i] in up:
                    return False
                pdict[pattern[i]] = seles[i]
                up.add(seles[i])

        return True

    # endregion

    # region Solution 434
    def countSegments(self, s: str) -> int:
        return len(s.split())

    # endregion

    # region Solution 509
    def fib(self, n: int) -> int:
        if n == 1:
            return 1
        if n == 0:
            return 0

        pre = 1
        sum = 1
        index = 2
        while index < n:
            sum += pre
            pre = sum
            index += 1

        return sum

    # endregion

    # region Solution 495
    def findPoisonedDuration(self, timeSeries: List[int], duration: int) -> int:
        sumdur = 0

        for i in range(1, len(timeSeries)):
            if timeSeries[i] - timeSeries[i - 1] >= duration:
                sumdur += duration
            else:
                sumdur += timeSeries[i] - timeSeries[i - 1]

        return sumdur + duration

    # endregion

    # region Solution 628
    def maximumProduct(self, nums: List[int]) -> int:
        nums.sort()
        maxnum = nums[len(nums) - 1] * nums[len(nums) - 2] * nums[len(nums) - 3]
        maxnum = max(maxnum, nums[0] * nums[1] * nums[-1])
        return maxnum

    # endregion

    # region Solution 1526
    def calcSumRange(
        self,
        root: SegmentTreeNode,
        posdict: Dict[int, List[int]],
        start: int,
        end: int,
        premin: int,
    ) -> int:
        val = SegmentTreeNode.query_segment_tree(root, start, end)
        if val <= premin or val == inf:
            return 0
        nextpos = posdict[val]
        tempstart = start
        tempsum = val - premin
        for i in range(len(nextpos)):
            if nextpos[i] - 1 >= start and nextpos[i] <= end:
                tempsum += self.calcSumRange(
                    root, posdict, tempstart, nextpos[i] - 1, val
                )
            if nextpos[i] >= start and nextpos[i] <= end:
                tempstart = nextpos[i] + 1
        if tempstart <= end:
            tempsum += self.calcSumRange(root, posdict, tempstart, end, val)
        return tempsum

    def minNumberOperations(self, target: List[int]) -> int:
        # 差分
        ans = pre = 0
        for x in target:
            if x < pre:
                ans += pre - x
            pre = x
        return ans + pre
        # totalsum = 0
        # root = SegmentTreeNode(0, len(target) - 1)
        # SegmentTreeNode.build_segment_tree(target, root, 0, len(target) - 1)
        # posdict: Dict[int, List[int]] = dict()
        # for i in range(len(target)):
        #     if target[i] not in posdict:
        #         posdict[target[i]] = list()
        #     posdict[target[i]].append(i)

        # totalsum = self.calcSumRange(root, posdict, 0, len(target) - 1, 0)
        # return totalsum

    # endregion

    # region Solution 361
    def maxKilledEnemies(self, grid: List[List[str]]) -> int:
        # dp[i][j] = dp[i-1][j] + dp[i+1][j] + dp[i][j-1] + dp[i][j+1]
        dp: List[List[Pos]] = [
            [Pos(0, 0, 0, 0) for _ in range(len(grid[0]))] for _ in range(len(grid))
        ]

        for r in range(len(grid)):
            for c in range(len(grid[0])):
                if grid[r][c] == "0" or grid[r][c] == "E":
                    if r - 1 >= 0:
                        if grid[r - 1][c] == "E":
                            dp[r][c].t = dp[r - 1][c].t + 1
                        elif grid[r - 1][c] == "W":
                            dp[r][c].t = 0
                        else:
                            dp[r][c].t = dp[r - 1][c].t
                    if c - 1 >= 0:
                        if grid[r][c - 1] == "E":
                            dp[r][c].l = dp[r][c - 1].l + 1
                        elif grid[r][c - 1] == "W":
                            dp[r][c].l = 0
                        else:
                            dp[r][c].l = dp[r][c - 1].l

        maxsum = 0
        for rr in range(len(grid) - 1, -1, -1):
            for rc in range(len(grid[0]) - 1, -1, -1):
                if grid[rr][rc] == "0" or grid[rr][rc] == "E":
                    if rr + 1 < len(grid):
                        if grid[rr + 1][rc] == "E":
                            dp[rr][rc].b = dp[rr + 1][rc].b + 1
                        elif grid[rr + 1][rc] == "W":
                            dp[rr][rc].b = 0
                        else:
                            dp[rr][rc].b = dp[rr + 1][rc].b
                    if rc + 1 < len(grid[0]):
                        if grid[rr][rc + 1] == "E":
                            dp[rr][rc].r = dp[rr][rc + 1].r + 1
                        elif grid[rr][rc + 1] == "W":
                            dp[rr][rc].r = 0
                        else:
                            dp[rr][rc].r = dp[rr][rc + 1].r
                    if grid[rr][rc] == "0":
                        maxsum = max(
                            maxsum,
                            dp[rr][rc].l + dp[rr][rc].t + dp[rr][rc].r + dp[rr][rc].b,
                        )

        return maxsum

    # endregion

    # region Solution 351
    def is_valid_351(self, current: int, pre: int, picked: list) -> bool:
        if pre == 0:
            return True
        if (current + pre) % 2 == 1:
            return True
        mid = (current + pre) // 2
        if mid == 4:
            return picked[mid]
        if ((current - 1) % 3 != (pre - 1) % 3) and (
            (current - 1) // 3 != (pre - 1) // 3
        ):
            return True
        return picked[mid]

    def dfs_351(
        self,
        m: int,
        n: int,
        picked: list,
        result: list,
        countpicked: int,
        current: int,
        pre: int,
    ):
        if m <= countpicked <= n:
            if not self.is_valid_351(current, pre, picked):
                return
            result[0] += 1
        if countpicked >= n:
            return
        for i in range(1, 10):
            if picked[i]:
                continue
            picked[i] = True
            countpicked += 1
            old = pre
            pre = current
            current = i
            self.dfs_351(m, n, picked, result, countpicked, current, pre)
            current = pre
            pre = old
            picked[i] = False
            countpicked -= 1

    def numberOfPatterns(self, m: int, n: int) -> int:
        totalsum = 0

        result = [0]
        picked = [False] * 10
        picked[1] = True
        self.dfs_351(m, n, picked, result, 1, 1, 0)
        totalsum += 4 * result[0]

        result[0] = 0
        picked = [False] * 10
        picked[2] = True
        self.dfs_351(m, n, picked, result, 1, 2, 0)
        totalsum += 4 * result[0]

        result[0] = 0
        picked = [False] * 10
        picked[5] = True
        self.dfs_351(m, n, picked, result, 1, 5, 0)
        totalsum += result[0]

        return totalsum

    # endregion

    # region Solution 357
    def countNumbersWithUniqueDigits(self, n: int) -> int:
        if n == 0:
            return 1
        return (
            factorial(10) // factorial(10 - n)
            - factorial(9) // factorial(10 - n)
            + self.countNumbersWithUniqueDigits(n - 1)
        )

    # endregion

    # region Solution 368
    def largestDivisibleSubset(self, nums: List[int]) -> List[int]:
        nums.sort()
        maxlength = 0
        result = None
        dp: List[List[int]] = [None for _ in range(len(nums))]
        for i in range(len(nums)):
            num = nums[i]
            temmax = -1
            maxindex = -1
            for j in range(i - 1, -1, -1):
                if num % nums[j] == 0:
                    if len(dp[j]) >= temmax:
                        maxindex = j
                        temmax = len(dp[j])
            if maxindex < 0:
                dp[i] = list([nums[i]])
            else:
                dp[i] = list(dp[maxindex])
                dp[i].append(num)
            length = len(dp[i])
            if length > maxlength:
                maxlength = length
                result = dp[i]

        return result

    # endregion

    # region Solution 473
    # TODO::重做
    def makesquare(self, matchsticks: List[int]) -> bool:
        return

    # endregion

    # region Solution 3289
    def getSneakyNumbers(self, nums: List[int]) -> List[int]:
        hashset = set()
        result = []
        for i in nums:
            if i in hashset:
                result.append(i)
            else:
                hashset.add(i)
        return result

    # endregion

    # region Solution 221
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        info: List[List[List[int]]] = [
            [[0] * 2 for _ in range(len(matrix[0]))] for _ in range(len(matrix))
        ]
        maxedge = 0
        for r in range(len(matrix)):
            for c in range(len(matrix[0])):
                if matrix[r][c] == "1":
                    maxedge = max(maxedge, 1)
                    info[r][c][1] = 1
                    info[r][c][0] = 1
                    if c > 0 and matrix[r][c - 1] == "1":
                        info[r][c][0] = info[r][c - 1][0] + 1
                    if r > 0 and matrix[r - 1][c] == "1":
                        minwidth = info[r][c][0]
                        minedge = maxedge
                        for i in range(2, info[r][c][0] + 1):
                            if r - i + 1 >= 0 and matrix[r - i + 1][c] == "1":
                                if minwidth < info[r - i + 1][c][1]:
                                    minedge = info[r - i + 1][c][1]
                                    break
                                minwidth = min(minwidth, info[r - i + 1][c][0])
                                if minwidth <= i:
                                    minedge = minwidth
                                    break
                            else:
                                minedge = min(i - 1, minwidth)
                                break
                        maxedge = max(maxedge, minedge)
                        info[r][c][1] = maxedge
        return maxedge * maxedge

    # endregion

    # region Solution 256
    def minCost(self, costs: List[List[int]]) -> int:
        for i in range(1, len(costs)):
            costs[i][0] = min(costs[i - 1][1], costs[i - 1][2]) + costs[i][0]
            costs[i][1] = min(costs[i - 1][0], costs[i - 1][2]) + costs[i][1]
            costs[i][2] = min(costs[i - 1][0], costs[i - 1][1]) + costs[i][2]
        return min(costs[len(costs) - 1])

    # endregion

    # region Solution 264
    def nthUglyNumber(self, n: int) -> int:
        dp = [0] * max(3, n)
        dp[0] = 1
        dp[1] = 2
        dp[2] = 3

        hashset = set()
        hashset.add(1)
        hashset.add(2)
        hashset.add(3)
        for i in range(3, n):
            minval = dp[i - 1]
            nextmin = inf
            for k in range(i):
                if dp[k] * 5 > minval:
                    if dp[k] * 3 > minval:
                        if dp[k] * 2 > minval:
                            if dp[k] * 2 not in hashset:
                                nextmin = min(nextmin, dp[k] * 2)
                            break
                        else:
                            if dp[k] * 3 not in hashset:
                                nextmin = min(nextmin, dp[k] * 3)
                    else:
                        if dp[k] * 5 not in hashset:
                            nextmin = min(nextmin, dp[k] * 5)
            hashset.add(nextmin)
            dp[i] = nextmin

        return dp[n - 1]

    # endregion

    # region Solution 474
    # 背包问题
    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        # dp [i][j] (i 填充 0 的个数) (j 填充 1 的个数)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for s in strs:
            count0 = s.count("0")
            count1 = s.count("1")
            for c1 in range(n, -1, -1):
                for c0 in range(m, -1, -1):
                    if c1 >= count1 and c0 >= count0:
                        dp[c0][c1] = max(dp[c0][c1], dp[c0 - count1][c1 - count0] + 1)
        return dp[m][n]

    # endregion

    # region Solution 3217
    ## 可以使用头部节点减少数值赋值操作
    def modifiedList(
        self, nums: List[int], head: Optional[ListNode]
    ) -> Optional[ListNode]:
        exist = set(nums)
        newhead = head
        current = head

        pre: Optional[ListNode] = None
        while current != None:
            if current.val in exist:
                if current.next != None:
                    current.val = current.next.val
                    current.next = current.next.next
                elif pre != None:
                    pre.next = None
                    current = None
                else:
                    return None
            else:
                pre = current
                current = current.next

        return newhead

    # endregion

    # region Solution 2257
    def countUnguarded(
        self, m: int, n: int, guards: List[List[int]], walls: List[List[int]]
    ) -> int:
        grid = [[0 for _ in range(n)] for _ in range(m)]
        remain = m * n - len(guards) - len(walls)
        for g in guards:
            grid[g[0]][g[1]] = 1
        for w in walls:
            grid[w[0]][w[1]] = 2

        for g in guards:
            if grid[g[0]][g[1]] != 3:
                for i in range(g[1], n):
                    if grid[g[0]][i] == 1:
                        grid[g[0]][i] = 3
                        continue
                    elif grid[g[0]][i] == 0:
                        grid[g[0]][i] = 3
                        remain -= 1
                    elif grid[g[0]][i] == 2:
                        break
                for i in range(g[1], -1, -1):
                    if grid[g[0]][i] == 1:
                        grid[g[0]][i] = 3
                        continue
                    elif grid[g[0]][i] == 0:
                        grid[g[0]][i] = 3
                        remain -= 1
                    elif grid[g[0]][i] == 2:
                        break
            if grid[g[0]][g[1]] != 4:
                for i in range(g[0], m):
                    if grid[i][g[1]] == 1:
                        grid[i][g[1]] = 4
                        continue
                    elif grid[i][g[1]] == 0:
                        grid[i][g[1]] = 4
                        remain -= 1
                    elif grid[i][g[1]] == 2:
                        break
                for i in range(g[0], -1, -1):
                    if grid[i][g[1]] == 1:
                        grid[i][g[1]] = 4
                        continue
                    elif grid[i][g[1]] == 0:
                        grid[i][g[1]] = 4
                        remain -= 1
                    elif grid[i][g[1]] == 2:
                        break

        return remain

    # endregion

    # region Solution 1578
    def minCost(self, colors: str, neededTime: List[int]) -> int:
        start = 0
        end = 0
        tempsum = neededTime[0]
        tempmax = neededTime[0]
        totalsum = 0
        for i in range(1, len(colors)):
            if colors[i] == colors[i - 1]:
                end = i
                tempsum += neededTime[i]
                tempmax = max(neededTime[i], tempmax)
            else:
                if end - start > 0:
                    totalsum += tempsum - tempmax
                start = end = i
                tempsum = neededTime[i]
                tempmax = neededTime[i]

        if end - start > 0:
            totalsum += tempsum - tempmax

        return totalsum

    # endregion

    # region Solution 270
    def search_270(self, root: Optional[TreeNode], target: float, result: list):
        if root is None:
            return
        pad1 = abs(target - root.val)
        pad3 = abs(target - result[0])
        if pad1 == pad3:
            result[0] = min(root.val, result[0])
        elif pad1 < pad3:
            result[0] = root.val
        if target > root.val:
            return self.search_270(root.right, target, result)
        else:
            return self.search_270(root.left, target, result)

    def closestValue(self, root: Optional[TreeNode], target: float) -> int:
        result = [root.val]
        self.search_270(root, target, result)
        return result[0]

    # endregion

    # region Solution 292
    def canWinNim(self, n: int) -> bool:
        return n % 4 != 0

    # endregion

    # region Solution 3318
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        occurdict = dict()
        result = [0] * (len(nums) - k + 1)
        for i in range(len(nums)):
            if nums[i] not in occurdict:
                occurdict[nums[i]] = 0
            occurdict[nums[i]] += 1
            if i >= k - 1:
                if i > k - 1:
                    occurdict[nums[i - k]] -= 1
                    if occurdict[nums[i - k]] == 0:
                        occurdict.pop(nums[i - k])
                tempresult = sorted(
                    occurdict.items(), key=lambda x: x[1] * 50 + x[0], reverse=True
                )
                for u in range(min(x, len(tempresult))):
                    result[i - (k - 1)] += tempresult[u][0] * tempresult[u][1]
        return result

    # endregion

    # region Solution 643
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        tempmax = -inf
        current_sum = sum(nums[:k])
        tempsum = current_sum
        lenn = len(nums)
        for i in range(k, lenn):
            tempsum = tempsum + nums[i] - nums[i - k]
            if tempsum > tempmax:
                tempmax = tempsum
        return tempmax / k

    # endregion

    # region Solution 645
    def findErrorNums(self, nums: List[int]) -> List[int]:
        nums.sort()
        result = [0] * 2
        if nums[0] != 1:
            result[1] = 1
        if nums[len(nums) - 1] != len(nums):
            result[1] = len(nums)
        for i in range(1, len(nums)):
            if nums[i] == nums[i - 1]:
                result[0] = nums[i]
            if nums[i] - nums[i - 1] > 1:
                result[1] = nums[i] - 1
        return result

    # endregion

    # region Solution 657
    def judgeCircle(self, moves: str) -> bool:
        result: Counter = Counter(moves)
        return result["L"] == result["R"] and result["U"] == result["D"]

    # endregion

    # region Solution 375
    # TODO::重做
    def getMoneyAmount(self, n: int) -> int:
        return

    # endregion
 # region Solution 276
    def numWays(self, n: int, k: int) -> int:
        # dp[n][k] = sum(dp[n-1][!k]) + if dp[n-2][k] > 0 ? 0 : dp[n-1][k]
        dp = [0] * n
        dp[0] = k
        for i in range(1, n):
            dp[i] = k * dp[i - 1]
            if i >= 2:
                dp[i] -= dp[i - 1]
                dp[i] += dp[i - 2] * (k - 1)
        return dp[n - 1]

    # endregion

    # region Solution 3321
    # TODO::
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        return

    # endregion
    
    # region Solution 294
    # TODO::
    def dfs_294(self, s:List[str]) -> bool:
        flag = False
        for i in range(1, len(s)):
            if s[i] == s[i-1] and s[i] == '+':
                s[i] = s[i-1] = '-'
                if not self.dfs_294(s):
                    flag = True
                    break
                s[i] = s[i-1] = '+'
        return flag
    
    def canWin(self, currentState: str) -> bool:
        return self.dfs_294(list(currentState)) 
    # endregion
    
    # region Solution 309
    # TODO::
    def maxProfit(self, prices: List[int]) -> int:
        ls = list()
        ls.append(prices[0])
        profit = 0
        lastpad = 0
        for i in range(1, len(prices)):
            num = prices[i]
            if num >= ls[-1]:
                ls.append(num)
            else:
                if lastpad > 0 and len(ls) > 1:
                    profit = max(profit - lastpad + ls[-1] - ls[0], profit + ls[-1] - ls[1])
                elif len(ls) > 0:
                    profit += ls[-1] - ls[0]
                if len(ls) > 1:
                    lastpad = ls[-1] - ls[-2]
                else:
                    lastpad = 0
                ls.clear()
                ls.append(num)

        if len(ls) > 1 and lastpad > 0:
            profit = max(profit - lastpad + ls[-1] - ls[0], profit + ls[-1] - ls[1])
        elif len(ls) > 0:
            profit += ls[-1] - ls[0]
        
        return profit
    # endregion
    
    # region Solution 337
    def rob_337(self, prepicked: bool, root: Optional[TreeNode], tempdict:Dict[Tuple[TreeNode,bool],int]):
        if root is None:
            return 0
        if prepicked:
            if (root.left, False) not in tempdict:
                tempdict[(root.left, False)] = self.rob_337(False, root.left, tempdict)
            if (root.right, False) not in tempdict:
                tempdict[(root.right, False)] = self.rob_337(False, root.right, tempdict)
            return tempdict[(root.left, False)] + tempdict[(root.right, False)]
        else:
            if (root.left, False) not in tempdict:
                tempdict[(root.left, False)] = self.rob_337(False, root.left, tempdict)
            if (root.right, False) not in tempdict:
                tempdict[(root.right, False)] = self.rob_337(False, root.right, tempdict)
            if (root.left, True) not in tempdict:
                tempdict[(root.left, True)] = self.rob_337(True, root.left, tempdict)
            if (root.right, True) not in tempdict:
                tempdict[(root.right, True)] = self.rob_337(True, root.right, tempdict)
            return max(
                root.val + tempdict[(root.left, True)] + tempdict[(root.right, True)],
                tempdict[(root.left, False)] + tempdict[(root.right, False)],
            )

    def rob(self, root: Optional[TreeNode]) -> int:
        # max(root + rob(0, root.left) + rob(0, root.right),0 + rob(1,root.left) + rob(1,root.right))
        tempdict = dict()
        return max(self.rob_337(False,root,tempdict),self.rob_337(True,root,tempdict))

    # endregion
