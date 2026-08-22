from sortedcontainers import SortedList


class Solution:

    def twoSum(self, nums: list[int], target: int) -> list[int]:
        aimset = dict()
        i = 0
        while i < len(nums):
            if nums[i] in aimset:
                return [aimset[nums[i]], i]
            aimset[(target - nums[i])] = i
            i += 1
        return []

    def groupAnagrams(self, strs: list[str]) -> list[list[str]]:
        aimset = dict()
        for s in strs:
            key = "".join(sorted(s))
            if key not in aimset:
                aimset[key] = []
            aimset[key].append(s)
        return list(aimset.values())

    def longestConsecutive(self, nums: list[int]) -> int:
        if len(nums) == 0:
            return 0
        numset = set(nums)
        result = 0
        for i in numset:
            if (i - 1) in numset:
                continue
            scount = 1
            j = 1
            while i + j in numset:
                scount += 1
                j += 1
            result = max(result, scount)
        return result

    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        start = 0
        offset = 1

        while True:
            if start + offset >= len(nums):
                break
            if nums[start] == 0:
                if nums[start + offset] == 0:
                    offset += 1
                else:
                    c = nums[start + offset]
                    nums[start] = c
                    nums[start + offset] = 0
                    start += 1
            else:
                start += 1

    def maxArea(self, height: List[int]) -> int:
        start = 0
        end = len(height) - 1

        result = 0
        while start < end:
            mheight = min(height[start], height[end])
            result = max(result, mheight * (end - start))
            if mheight == height[start]:
                start += 1
            elif mheight == height[end]:
                end -= 1
        return result

    def threeSum(self, nums: list[int]) -> list[list[int]]:
        sumdict: dict[int, list[int]] = dict()
        for i in range(len(nums)):
            if nums[i] not in sumdict:
                sumdict[nums[i]] = []
            sumdict[nums[i]].append(i)
        dictlist = list(sumdict)
        pass

    def lengthOfLongestSubstring(self, s: str) -> int:
        if len(s) == 0:
            return 0
        start = 0
        charset = [-1] * 256
        charset[ord(s[0])] = 0
        maxlen = 0
        for i in range(1, len(s)):
            cpos = ord(s[i])
            if charset[cpos] >= 0:
                maxlen = max(maxlen, i - start)
                start = max(start, charset[cpos] + 1)
            charset[cpos] = i
        maxlen = max(maxlen, len(s) - start)
        return maxlen
