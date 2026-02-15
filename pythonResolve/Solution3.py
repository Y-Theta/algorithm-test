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