class Solution:
    def sumGame(self, num: str) -> bool:
        presum = 0
        precount = 0
        postsum = 0
        postcount = 0
        for i in range(len(num)):
            if i < len(num) // 2:
                if num[i] == '?':
                    precount += 1
                    presum += 0
                    continue
                presum += ord(num[i]) - ord('0')
            else:
                if num[i] == '?':
                    postcount += 1
                    postsum += 0
                    continue
                postsum += ord(num[i]) - ord('0')
        
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
        for i , c in enumerate(s):
            if c == '1':
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
                        result.append(s[kpos[0]:kpos[-1] + 1])
        if len(result) == 0:
            return ""
        result.sort()
        return result[0]
