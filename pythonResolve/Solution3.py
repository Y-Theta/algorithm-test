from Common import ListNode, TreeNode, SegmentTreeNode, Pos, UniFind,Pt
from typing import Optional, List, Dict, Counter, Tuple, Set
from math import gcd, sqrt, inf, factorial
from dataclasses import dataclass
import re
import numpy as np

class Solution3:

    # region Solution 651
    def maxA(self, n: int) -> int:
        if n <= 6:
            return n

        dp = [0 for _ in range(n)]
        dp[0] = 1

        for i in range(1, n):
            dp[i] = dp[i-1] + 1
            if i >= 3:
                tempmax = dp[i - 3] * 2
                k = 4
                while i - k >= 0:
                    if dp[i - k] * (k - 1) < tempmax:
                        break
                    tempmax = dp[i - k] *  (k -1 )
                    k += 1
                dp[i] = max(dp[i][0], tempmax)

        return dp[n - 1]

    # endregion
    
    # region Solution 67
    def addBinary(self, a: str, b: str) -> str:
        la = list(a)
        la.reverse()
        lb = list(b)
        lb.reverse()
        sresult = []
        index = 0
        presum = 0
        while index < len(la) or index < len(lb):
            ca = 0
            if index < len(la):
                ca = ord(la[index]) - ord('0')
            cb = 0
            if index < len(lb):
                cb = ord(lb[index]) - ord('0')
            csum = ca + cb + presum
            presum = csum // 2
            current = csum % 2
            sresult.insert(0,str(current))
            index+=1
        
        if presum > 0:
            sresult.insert(0,'1')
        
        return ''.join(sresult)
        
    # endregion
    
    # region Solution 422
    def validWordSquare(self, words: List[str]) -> bool:
        for i in range(len(words)):
            if len(words) < len(words[i]):
                return False
            for j in range(i + 1):
                if j < len(words[i]):
                    if len(words[j]) <= i or words[i][j] != words[j][i]:
                        return False
                elif len(words[j]) > i:
                    return False
        return True
    # endregion
    
    # region Solution 1689
    def minPartitions(self, n: str) -> int:
        return max(n) - ord('0')
    # endregion

    # region Solution 582
    def killProcess(self, pid: List[int], ppid: List[int], kill: int) -> List[int]:
        procmap = dict()
        for i in range(len(pid)):
            _id = pid[i]
            p_id = ppid[i]
            if p_id not in procmap:
                procmap[p_id] = set()
            procmap[p_id].add(_id)
        
        to_kill = list()
        to_kill.append(kill)
        if kill not in procmap:
            return to_kill
        
        to_delete = procmap[kill]
        new_to_delete = set()
        while len(to_delete) > 0:
            for i in to_delete:
                to_kill.append(i)
                if i in procmap:
                    for k in procmap[i]:
                        new_to_delete.add(k)
            to_delete = list(new_to_delete)
            new_to_delete.clear()

        return to_kill
    # endregion

    # region Solution 2087
    def minCost(self, startPos: List[int], homePos: List[int], rowCosts: List[int], colCosts: List[int]) -> int:
        xoffset = startPos[1] - homePos[1]
        yoffset = startPos[0] - homePos[0]
        
        cost = 0
        if xoffset != 0:
            if xoffset < 0:
                for i in range(startPos[1] + 1, homePos[1] + 1):
                    cost += colCosts[i]
            else:
                for i in range(homePos[1], startPos[1]):
                    cost += colCosts[i]

        if yoffset != 0:
            if yoffset < 0:
                for i in range(startPos[0] + 1, homePos[0] + 1):
                    cost += rowCosts[i]
            else:
                for i in range(homePos[0], startPos[0]):
                    cost += rowCosts[i]

        return cost
    # endregion

    # region Solution 874
    def robotSim(self, commands: List[int], obstacles: List[List[int]]) -> int:
        xobmap:Dict[int,List[int]] = dict()
        yobmap:Dict[int,List[int]] = dict()
        for i in obstacles:
            if i[0] not in xobmap:
                xobmap[i[0]] = list()
            xobmap[i[0]].append(i[1])
            if i[1] not in yobmap:
                yobmap[i[1]] = list()
            yobmap[i[1]].append(i[0])
        
        for item in xobmap.keys():
            xobmap[item] = sorted(xobmap[item])
        
        for item in yobmap.keys():
            yobmap[item] = sorted(yobmap[item])

        dir = 90
        yoffset = 0
        xoffset = 0
        maxoffset = 0
        for c in commands:
            if c == -2:
                dir = (dir + 90) % 360
            elif c == -1:
                dir = (dir - 90) % 360
            else:
                if dir == 90:
                    yoffsettemp = yoffset + c
                    if xoffset in xobmap:
                        for i in xobmap[xoffset]:
                            if i > yoffset and i <= yoffset + c:
                                yoffsettemp = i - 1
                                break
                    yoffset = yoffsettemp
                elif dir == 0:
                    xoffsettemp = xoffset + c
                    if yoffset in yobmap:
                        for i in yobmap[yoffset]:
                            if i > xoffset and i <= xoffset + c:
                                xoffsettemp = i - 1
                                break
                    xoffset = xoffsettemp
                elif dir == 180:
                    xoffsettemp = xoffset - c
                    if yoffset in yobmap:
                        for i in yobmap[yoffset]:
                            if i >= xoffset - c and i < xoffset:
                                xoffsettemp = i + 1
                                break
                    xoffset = xoffsettemp
                elif dir == 270:
                    yoffsettemp = yoffset - c
                    if xoffset in xobmap:
                        for i in xobmap[xoffset]:
                            if i >= yoffset - c and i < yoffset:
                                yoffsettemp = i + 1
                                break
                    yoffset = yoffsettemp
                maxoffset = max(maxoffset, xoffset * xoffset + yoffset * yoffset)
        return maxoffset
    # endregion

    # region Solution 163
    def findMissingRanges(self, nums: List[int], lower: int, upper: int) -> List[List[int]]:
        miss = list()
        if len(nums) == 0:
            miss.append([lower,upper])
            return miss
        if nums[0] > lower:
            miss.append([lower,nums[0] - 1])
        for i in range(1, len(nums)):
            if nums[i] - nums[i-1] > 1:
                miss.append([nums[i-1] + 1,nums[i] - 1])
        if nums[-1] < upper:
            miss.append([nums[-1] + 1,upper])
        return miss;
    # endregion
    
    # region Solution 1855
    def find_1855(self, nums1:List[int], start: int, end:int ,aim:int) -> int:
        while start < end:
            mid = (start + end) // 2
            if nums1[mid] > aim:
                start = mid + 1
            elif nums1[mid] <= aim:
                end = mid
        return (start + end) // 2
    
    def maxDistance(self, nums1: List[int], nums2: List[int])-> int:
        n2max = len(nums2)
        
        lenmax = 0 
        for i in range(n2max - 1, -1 , -1):
            aim = nums2[i]
            maxmatch = self.find_1855(nums1,0,len(nums1) - 1,aim)
            if aim >= nums1[maxmatch]:
                lenmax = max(lenmax, i - maxmatch)
                if lenmax >= n2max - 1:
                    return lenmax
        return lenmax
    # endregion
    
    # region Solution 877
    def stoneGame(self, piles: List[int]) -> bool:
        lenpiles = len(piles)
        dp = [[0 for _ in range(lenpiles)] for _ in range(lenpiles)]
        
        for i in range(lenpiles):
            dp[i][i] = piles[i]
            
        for i in range(lenpiles - 2 ,-1,-2):
            for j in range(i, lenpiles):
                dp[i][j] = max(piles[i] - dp[i+1][j], piles[j] - dp[i][j-1])   
        
        return dp[0][lenpiles - 1]
    # endregion
    
    # region 758
    def boldWords(self, words: List[str], s: str) -> str:
        wordsdic:Dict[str,List[str]] = dict()
        words.sort(key=len,reverse= True)
        
        for word in words:
            if word[0] not in wordsdic:
                wordsdic[word[0]] = []
            wordsdic[word[0]].append(word)
            
        pose = -1
        laste = -1
        result = []
        for i in range(len(s)):
            if s[i] in wordsdic:
                for k in wordsdic[s[i]]:
                    if s[i: i + len(k)] == k:
                        pose = max(pose, i + len(k))
                        if i > laste:
                            result.append("<b>")
                        break
            if i == pose:
                result.append("</b>")
            result.append(s[i])
            laste = pose
        if pose == len(s):
            result.append("</b>")
        return "".join(result)
    # endregion
    
    # region Solution 1081
    def smallestSubsequence(self, s: str) -> str:
        strdic:Dict[str, int] = dict()
        for i in range(len(s)):
            strdic[s[i]] = i
        stack = [] 
        
        stack.append(s[0])
        for i in range(1, len(s)):
            if s[i] in stack:
                continue
            while len(stack) > 0 and s[i] <= stack[-1]:
                if strdic[stack[-1]] <= i:
                    break
                stack.pop()
            stack.append(s[i])
        return "".join(stack)
                        
    # endregion
    
    # region Solution 1260
    def shiftGrid(self, grid: List[List[int]], k: int) -> List[List[int]]:
        row = len(grid)
        col = len(grid[0])
        colturn = k // col
        coltimes = k % col
        rowturn = colturn % row
        
        result = [None] * row 
        for i in range(row):
            result[i] = grid[(row + i - rowturn) % row]
        
        newresult = [[0 for _ in range(col)] for _ in range(row)]
        for c in range(col):
            for r in range(row):
                if c < coltimes:
                    newresult[r][c] = result[r-1][c - coltimes]
                else:
                    newresult[r][c] = result[r][c-coltimes]
        return newresult
    # endregion
    # region Solution 3514
    def uniqueXorTriplets(self, nums: List[int]) -> int:
        maxval = max(nums)
        digit = 0
        while maxval > 0:
            maxval //= 2
            digit += 1        
        
        result1 = [False] * (2 ** (digit + 1))
        for i in range(len(nums)):
            for j in range(len(nums)):
                result1[nums[i] ^ nums[j]] = True
        
        result2 = [False] * len(result1)
        for i in range(len(result1)):
            for j in range(len(nums)):
                result2[i ^ nums[j]] = True
        
        cr2 = Counter(result2)
        return cr2[True]
    # endregion
    
    # region Solution 3536
    def maxProduct(self, n: int) -> int:
        digit = n % 10
        maxreult = digit
        maxtemp = digit
        while (n // 10) > 0:
            n //= 10
            digit = n % 10
            maxreult = max(maxreult, maxtemp * digit) 
            maxtemp = max(digit, maxtemp)
        return maxreult
    # endregion