from Common import ListNode, TreeNode, SegmentTreeNode, Pos, UniFind
from typing import Optional, List, Dict, Counter, Tuple
from sortedcontainers import SortedList
from math import gcd, sqrt, inf, factorial
from dataclasses import dataclass


class Solution2:
    # region Solution 1513
    def numSub(self, s: str) -> int:
        if s.count("1") == 0:
            return 0

        # presum = [0] * len(s)
        # presum[0] = 1 if s[0] == "1" else 0
        # for i in range(1, len(s)):
        #     presum[i] = presum[i - 1] if s[i] == "0" else presum[i - 1] + 1

        last0pos = -1
        dp = 1 if s[0] == "1" else 0
        if s[0] == "0":
            last0pos = 0

        for i in range(1, len(s)):
            if s[i] == "0":
                last0pos = i
            else:
                dp = dp + (i - last0pos)

        return dp % (10**9 + 7)

    # endregion

    # region Solution 717
    def isValid_717(self, bits: List[int], index: int):
        flag = True
        if index < 0:
            return flag
        if index >= 0:
            flag = bits[index] == 0 and self.isValid_717(bits, index - 1)
        if index - 1 >= 0:
            flag = flag or bits[index - 1] == 1 and self.isValid_717(bits, index - 2)
        return flag

    def isOneBitCharacter(self, bits: List[int]) -> bool:
        if len(bits) <= 2:
            return bits[0] == 0
        flag = bits[len(bits) - 2] == 1 and self.isValid_717(bits, len(bits) - 3)
        return not flag

    # endregion

    # region Solution 2154
    def findFinalValue(self, nums: List[int], original: int) -> int:
        numsset = set(nums)

        while original in numsset:
            numsset.remove(original)
            original = 2 * original

        return original

    # endregion

    # region Solution 3190
    def minimumOperations(self, nums: List[int]) -> int:
        opt = 0
        for i in nums:
            if i % 3 == 0:
                continue
            opt += min(i % 3, 3 - (i % 3))
        return opt

    # endregion

    # region Solution 1262
    def maxSumDivThree(self, nums: List[int]) -> int:
        basesum = 0
        nv = []
        for i in range(len(nums)):
            if nums[i] % 3 == 0:
                basesum += nums[i]
            else:
                nv.append(nums[i])

        dp = [[0] * 3 for _ in range(len(nv))]
        firstmod3 = nv[0] % 3
        dp[0][firstmod3] = nv[0]

        for i in range(1, len(nv)):
            for j in range(3):
                nnv = nv[i] + dp[i - 1][j]
                nnvmod3 = nnv % 3
                dp[i][nnvmod3] = max(dp[i][nnvmod3], dp[i - 1][nnvmod3], nnv)
            for j in range(3):
                dp[i][j] = max(dp[i][j], dp[i - 1][j])

        return basesum + dp[len(nv) - 1][0]

    # endregion

    # region Solution 1018
    def prefixesDivBy5(self, nums: List[int]) -> List[bool]:
        result = [False] * len(nums)
        num = 0
        for i in range(len(nums)):
            num = nums[i] + (num << 1)
            result[i] = num % 5 == 0
        return result

    # endregion

    # region Solution 1015
    def smallestRepunitDivByK(self, k: int) -> int:
        kk = k
        digit = 0
        base = 0
        while kk > 0:
            digit += 1
            base = base * 10 + 1
            kk = kk // 10

        if base < k:
            base = base * 10 + 1
            digit += 1

        modset = set()
        modk = base % k
        if modk == 0:
            return digit

        while modk != 0:
            if modk in modset:
                return -1
            else:
                modset.add(modk)
                modk = (modk * 10 + 1) % k
                digit += 1

        return digit

    # endregion

    # region Solution 2435
    def numberOfPaths(self, grid: List[List[int]], k: int) -> int:
        dp = [
            [[0 for _ in range(k)] for _ in range(len(grid[0]))]
            for _ in range(len(grid))
        ]

        dp[0][0][grid[0][0] % k] = 1

        for i in range(1, len(grid)):
            for m in range(k):
                if dp[i - 1][0][m] > 0:
                    modk = (m + grid[i][0]) % k
                    dp[i][0][modk] = 1

        for j in range(1, len(grid[0])):
            for m in range(k):
                if dp[0][j - 1][m] > 0:
                    modk = (m + grid[0][j]) % k
                    dp[0][j][modk] = 1

        for i in range(1, len(grid)):
            for j in range(1, len(grid[0])):
                for m in range(k):
                    if dp[i - 1][j][m] > 0 or dp[i][j - 1][m] > 0:
                        modk = (m + grid[i][j]) % k
                        dp[i][j][modk] = dp[i - 1][j][m] + dp[i][j - 1][m]

        return dp[len(grid) - 1][len(grid[0]) - 1][0] % ((10**9) + 7)

    # endregion

    # region Solution 3512
    def minOperations(self, nums: List[int], k: int) -> int:
        total = sum(nums)
        return total % k

    # endregion

    # region Solution 1590
    def minSubarray(self, nums: List[int], p: int) -> int:
        presum = [0] * len(nums)
        mincount = [-1] * p
        presum[0] = nums[0] % p
        flag = False
        
        for i in range(1, len(nums)):
            presum[i] = presum[i - 1] + (nums[i] % p)
            
        totalmod = presum[len(nums) - 1] % p
        if totalmod == 0:
            return 0
        
        return mincount[totalmod]

    # endregion
