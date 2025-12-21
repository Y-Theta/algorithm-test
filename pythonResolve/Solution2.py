from Common import ListNode, TreeNode, SegmentTreeNode, Pos, UniFind
from typing import Optional, List, Dict, Counter, Tuple
from math import gcd, sqrt, inf, factorial
from dataclasses import dataclass
import re


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
        total = sum(nums)
        totalmod = total % p
        if totalmod == 0:
            return 0

        mindis = len(nums)
        moddic2 = dict()
        presum = [0] * len(nums)
        presum[0] = nums[0]
        moddic2[0] = -1
        moddic2[presum[0] % p] = 0
        for i in range(1, len(nums)):
            presum[i] = presum[i - 1] + nums[i]
            presummodp = presum[i] % p
            key = presummodp - totalmod
            if key < 0:
                key += p
            if key in moddic2:
                mindis = min(mindis, i - moddic2[key])
            moddic2[presummodp] = i

        return -1 if mindis == len(nums) else mindis

    # endregion

    # region Solution 3623
    def countTrapezoids(self, points: List[List[int]]) -> int:
        rowdic = dict()
        for pt in points:
            if pt[1] not in rowdic:
                rowdic[pt[1]] = 1
            else:
                rowdic[pt[1]] += 1

        totalsum = 0
        countlist = list(rowdic.items())
        dp = [0] * len(countlist)
        dp[0] = countlist[0][1] * (countlist[0][1] - 1) // 2
        for i in range(1, len(countlist)):
            dp[i] = countlist[i][1] * (countlist[i][1] - 1) // 2
            for j in range(i):
                totalsum += dp[i] * dp[j]
        return totalsum % ((10**9) + 7)

    # endregion

    # region Solution 326
    def isPowerOfThree(self, n: int) -> bool:
        if n == 0:
            return False
        if n == 1:
            return True
        if n % 3 == 0:
            return self.isPowerOfThree(n // 3)
        return False

    # endregion

    # region Solution 342
    def isPowerOfFour(self, n: int) -> bool:
        # 10 10000
        if n == 0:
            return False
        if n == 1:
            return True
        if n & (n - 1) != 0:
            return False
        if n == 2 or n == 8:
            return False
        s = int(sqrt(abs(n)))
        return s & (s - 1) == 0

    # endregion

    # region Solution 367
    def isPerfectSquare(self, num: int) -> bool:
        for i in range(num + 1):
            if i * i == num:
                return True
            if i * i > num:
                return False
        return False

    # endregion

    # region Solution 3432
    def countPartitions(self, nums: List[int]) -> int:
        total = 0
        for i in range(len(nums) - 1):
            desc = sum(nums[0:i]) - sum(nums[i + 1 : len(nums)])
            if desc % 2 == 0:
                total += 1
        return total

    # endregion

    # region Solution 3578

    # def countPartitions(self, nums: List[int], k: int) -> int:
    # dp = [0] * (len(nums))
    # dp[0] = 1
    # for i in range(1, len(nums)):
    #     tempsum = dp[i-1]
    #     tempmax = tempmin = nums[i]
    #     for j in range(i - 1 , -1, -1):
    #         tempmax = max(tempmax, nums[j])
    #         tempmin = min(tempmin, nums[j])
    #         if tempmax - tempmin > k:
    #             break
    #         tempsum += dp[j - 1]
    #     if j == 0 and tempmax - tempmin <= k:
    #         tempsum += 1
    #     dp[i] = tempsum
    # return dp[len(nums) - 1] % ((10 ** 9) + 7)

    def countPartitions(self, nums: List[int], k: int) -> int:
        dp = [0] * (len(nums))
        dp[0] = 1
        qmax = []
        qmin = []
        qmax.append(nums[0])
        qmin.append(nums[0])
        for i in range(1, len(nums)):
            tempsum = dp[i - 1]
            if nums[i] > qmax[-1]:
                qmax.append(nums[i])
            if nums[i] < qmin[-1]:
                qmin.append(nums[i])
            for j in range(i - 1, -1, -1):
                tempmax = max(tempmax, nums[j])
                tempmin = min(tempmin, nums[j])
                if tempmax - tempmin > k:
                    break
                tempsum += dp[j - 1]
            if j == 0 and tempmax - tempmin <= k:
                tempsum += 1
            dp[i] = tempsum
        return dp[len(nums) - 1] % ((10**9) + 7)

    # endregion

    # region Solution 1523
    def countOdds(self, low: int, high: int) -> int:
        countodd = (high - low) // 2
        if low % 2 == 1:
            countodd += 1
        elif high % 2 == 1:
            countodd += 1
        return countodd

    # endregion

    # region Solution 344
    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        start = 0
        end = len(s) - 1
        while start < end:
            c = s[end]
            s[end] = s[start]
            s[start] = c
            end -= 1
            start += 1

    # endregion

    # region Solution 345
    def reverseVowels(self, string: str) -> str:
        s = list(string)
        start = 0
        end = len(s) - 1
        textset = set({"a", "e", "i", "o", "u", "A", "E", "I", "O", "U"})
        while start < end:
            forward = s[start] in textset
            backward = s[end] in textset
            if not forward:
                start += 1
                continue
            if not backward:
                end -= 1
                continue
            c = s[end]
            s[end] = s[start]
            s[start] = c
            end -= 1
            start += 1
        return "".join(s)

    # endregion

    # region Solution 1925
    def countTriples(self, n: int) -> int:
        totalnum = 0
        powset = set()
        for i in range(1, n + 1):
            powset.add(i**2)
        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                sqrtnum = sqrt(i**2 + j**2)
                if sqrtnum in powset:
                    totalnum += 2
        return totalnum

    # endregion

    # region Solution 298
    def dfs_298(self, root: Optional[TreeNode], currentsum: int) -> int:
        if root is None:
            return currentsum
        currentmax = currentsum
        if root.left != None:
            if root.left.val - root.val == 1:
                currentmax = max(currentmax, self.dfs_298(root.left, currentsum + 1))
            else:
                currentmax = max(currentmax, self.dfs_298(root.left, 1))

        if root.right != None:
            if root.right.val - root.val == 1:
                currentmax = max(currentmax, self.dfs_298(root.right, currentsum + 1))
            else:
                currentmax = max(currentmax, self.dfs_298(root.right, 1))

        return currentmax

    def longestConsecutive(self, root: Optional[TreeNode]) -> int:
        return self.dfs_298(root, 1)

    # endregion
    # region Solution 3531
    def countCoveredBuildings(self, n: int, buildings: List[List[int]]) -> int:
        totalsum = 0
        return totalsum

    # endregion

    # region Solution 3606
    def validateCoupons(
        self, code: List[str], businessLine: List[str], isActive: List[bool]
    ) -> List[str]:
        result1 = []
        result2 = []
        result3 = []
        result4 = []
        pattern = re.compile(r"^[a-zA-z0-9_]+$")
        for i in range(len(code)):
            if pattern.match(code[i]) != None and isActive[i]:
                if businessLine[i] == "electronics":
                    result1.append(code[i])
                if businessLine[i] == "grocery":
                    result2.append(code[i])
                if businessLine[i] == "pharmacy":
                    result3.append(code[i])
                if businessLine[i] == "restaurant":
                    result4.append(code[i])
        result = []
        result1.sort()
        result2.sort()
        result3.sort()
        result4.sort()
        result += result1
        result += result2
        result += result3
        result += result4
        return result

    # endregion

    # region Solution 3583

    def binary_search_index(self, arr: List[int], aim: int, start: int, end: int):
        while start < end:
            mid = (start + end) // 2
            if arr[mid] >= aim:
                end = mid 
            else:
                start = mid + 1
        mid = (start + end) // 2
        return mid

    def specialTriplets(self, nums: List[int]) -> int:
        numsdict: Dict[int, List[int]] = dict()
        for i in range(len(nums)):
            num = nums[i]
            if num not in numsdict:
                numsdict[num] = []
            numsdict[num].append(i)

        total = 0
        keys = sorted(numsdict.keys())
        for i in range(len(keys)):
            aimkey = keys[i] * 2
            if aimkey in numsdict:
                aimarr = numsdict[aimkey]
                for j in numsdict[keys[i]]:
                    if aimarr[0] < j and aimarr[len(aimarr) - 1] > j:
                        pos = self.binary_search_index(aimarr, j, 0, len(aimarr) - 1)
                        offset = 0
                        if aimkey == 0:
                            offset = 1
                        left = pos + offset
                        right = len(aimarr) - (pos + offset)
                        if aimkey == 0:
                            left -= 1                            
                        total += max(0, left * right)

        return total % ((10**9) + 7)

    # endregion
    
    # region Solution 944
    def minDeletionSize(self, strs: List[str]) -> int:
        len1 = len(strs[0])
        lentotal = len(strs)
        if lentotal == 1:
            return 0
        
        result = 0
        for j in range(len1):
            for i in range(1, lentotal):
                if strs[i][j] < strs[i - 1][j]:
                    result += 1
                    break
        
        return result
    # endregion
    
    # region Solution 955
    def minDeletionSize(self, strs: List[str]) -> int:
        len1 = len(strs[0])
        lentotal = len(strs)
        if lentotal == 1:
            return 0
        
        for j in range(len1):
            for i in range(1, lentotal):
                if strs[i][j] < strs[i - 1][j]:
                    break
        
        return
    # endregion