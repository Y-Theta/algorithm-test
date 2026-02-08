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
