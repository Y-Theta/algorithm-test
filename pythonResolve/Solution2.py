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
