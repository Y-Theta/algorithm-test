from Common import ListNode, TreeNode, SegmentTreeNode, Pos, UniFind
from typing import Optional, List, Dict, Counter, Tuple
from math import gcd, sqrt, inf, factorial
from dataclasses import dataclass
import re


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