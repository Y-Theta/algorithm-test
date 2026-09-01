from sortedcontainers import SortedList
from collections import Counter, defaultdict


class Solution:
    def sumGame(self, num: str) -> bool:
        presum = 0
        precount = 0
        postsum = 0
        postcount = 0
        for i in range(len(num)):
            if i < len(num) // 2:
                if num[i] == "?":
                    precount += 1
                    presum += 0
                    continue
                presum += ord(num[i]) - ord("0")
            else:
                if num[i] == "?":
                    postcount += 1
                    postsum += 0
                    continue
                postsum += ord(num[i]) - ord("0")

        cursumdes = presum - postsum
        countdes = precount - postcount
        if abs(countdes) % 2 == 1:
            return True

        if countdes == 0 and cursumdes == 0:
            return False
        if countdes < 0:
            return cursumdes != 9 * abs(countdes) // 2
        if countdes > 0:
            return cursumdes != -9 * abs(countdes) // 2

        return True

    def stoneGameVIII(self, stones: List[int]) -> int:
        nst = len(stones)
        prefix = [0] * nst
        prefix[0] = stones[0]
        for i in range(1, nst):
            prefix[i] = prefix[i - 1] + stones[i]

        dp = prefix[-1]
        for i in range(len(stones) - 2, 0, -1):
            dp = max(dp, prefix[i] - dp)
        return dp[1]

    def missingMultiple(self, nums: List[int], k: int) -> int:
        hashset = set(nums)
        i = 1
        while True:
            if i * k in hashset:
                i += 1
                continue
            return i * k

    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        kpos = []
        ki = 0
        rlen = len(s)
        result = []
        for i, c in enumerate(s):
            if c == "1":
                if ki < k:
                    kpos.append(i)
                    ki += 1
                else:
                    kpos.pop(0)
                    kpos.append(i)
                if ki == k:
                    nlen = kpos[-1] - kpos[0] + 1
                    if len(result) == 0 or len(result[0]) >= nlen:
                        if len(result) > 0 and len(result[0]) > nlen:
                            result.clear()
                        result.append(s[kpos[0] : kpos[-1] + 1])
        if len(result) == 0:
            return ""
        result.sort()
        return result[0]

    def similarRGB(self, color: str) -> str:
        pair = SortedList([i + 16 * i for i in range(16)])
        result = "#"
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        idx = pair.bisect_left(r)
        if idx >= len(pair) or pair[idx] == r:
            result += format(r, "02x")
        else:
            num = pair[idx]
            if idx > 0 and r - pair[idx - 1] < pair[idx] - r:
                num = pair[idx - 1]
            result += format(num, "02x")

        idx = pair.bisect_left(g)
        if idx >= len(pair) or pair[idx] == g:
            result += format(g, "02x")
        else:
            num = pair[idx]
            if idx > 0 and g - pair[idx - 1] < pair[idx] - g:
                num = pair[idx - 1]
            result += format(num, "02x")

        idx = pair.bisect_left(b)
        if idx >= len(pair) or pair[idx] == b:
            result += format(b, "02x")
        else:
            num = pair[idx]
            if idx > 0 and b - pair[idx - 1] < pair[idx] - b:
                num = pair[idx - 1]
            result += format(num, "02x")

        return result

    def lexPalindromicPermutation(self, s: str, target: str) -> str:
        sc = Counter(s)
        oddcount = 0
        oddc = ""
        for i in sc:
            if sc[i] % 2 == 1:
                oddcount += 1
                oddc = i
                if oddcount > 1:
                    return ""
        cdict = SortedList()
        for i in sc:
            for j in range(sc[i] // 2):
                cdict.add(i)

        halftarget = len(target) // 2
        if halftarget == 0:
            if ord(s) > ord(target):
                return s
            return ""
        clen = len(cdict)
        finalstr = None
        result = []

        def dfs(start: int = 0, limit: bool = True) -> bool:
            nonlocal finalstr
            if start == clen:
                if not limit:
                    finalstr = "".join(result)
                    finalstr = finalstr + oddc + finalstr[::-1]
                    return True
                if oddc != "":
                    if ord(oddc) > ord(target[halftarget]):
                        finalstr = "".join(result)
                        finalstr = finalstr + oddc + finalstr[::-1]
                        return True
                    elif ord(oddc) < ord(target[halftarget]):
                        return False
                for i in range(halftarget):
                    if result[-i - 1] < target[i + halftarget]:
                        return False
                    elif result[-i - 1] > target[i + halftarget]:
                        finalstr = "".join(result)
                        finalstr = finalstr + oddc + finalstr[::-1]
                        return True
                return False
            if not limit:
                for i in range(len(cdict)):
                    result.append(cdict[0])
                finalstr = "".join(result)
                finalstr = finalstr + oddc + finalstr[::-1]
                return True
            for i in range(ord("z") - ord(target[start]) + 1):
                aimchar = ord(target[start]) + i
                cid = cdict.bisect_left(chr(aimchar))
                if cid >= len(cdict):
                    return False
                findc = cdict[cid]
                cdict.pop(cid)
                result.append(findc)
                if dfs(start + 1, i == 0):
                    return True
                cdict.add(findc)
                result.pop(-1)

        if not dfs():
            return ""
        return finalstr

    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        valwithpos = [(v,i) for i,v in enumerate(nums)]
        valordlist = sorted(valwithpos, key=lambda x:x[0])

        start = 0
        sortedpos = SortedList()
        sortedpos.add(valordlist[0][1])
        for i in range(1, len(valordlist)):
            if valordlist[i][0] - valordlist[i - 1][0] > limit:
                for j in range(len(sortedpos)):
                    nums[sortedpos[j]] = valordlist[start + j][0]
                start = i
                sortedpos.clear()
            sortedpos.add(valordlist[i][1])
        if len(sortedpos) > 0:
            for j in range(len(sortedpos)):
                nums[sortedpos[j]] = valordlist[start + j][0]
        
        return nums

