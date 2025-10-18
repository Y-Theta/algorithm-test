using System;
using System.Collections.Generic;
using System.Linq;

namespace main.csharpResolve
{
    internal class ResolveClass1
    {
        #region   Solution 3494
        public long Solution_3494(int[] skill, int[] mana)
        {
            long[] lasttime = new long[skill.Length];
            lasttime[0] = skill[0] * mana[0];
            for (int i = 1; i < skill.Length; i++)
            {
                lasttime[i] = lasttime[i - 1] + (skill[i] * mana[0]);
            }
            long nextstart = 0;
            for (int i = 1; i < mana.Length; i++)
            {
                var current = mana[i];
                long timecost = 0;
                for (int j = 0; j < skill.Length; j++)
                {
                    if (j > 0)
                    {
                        nextstart = Math.Max(nextstart, lasttime[j] - timecost);
                    }
                    else
                    {
                        nextstart = Math.Max(nextstart, lasttime[j]);
                    }
                    timecost += current * skill[j];
                }
                lasttime[0] = nextstart + current * skill[0];
                for (int k = 1; k < lasttime.Length; k++)
                {
                    lasttime[k] = lasttime[k - 1] + (current * skill[k]);
                }
                nextstart = 0;
            }

            return lasttime.Last();
        }
        #endregion

        #region   Solution 20
        public bool Solution_0020(string s)
        {
            Stack<char> a = new Stack<char>();
            Dictionary<char, char> pairmap = new Dictionary<char, char>();
            pairmap['{'] = '}';
            pairmap['['] = ']';
            pairmap['('] = ')';
            for (int i = 0; i < s.Length; i++)
            {
                var ch = s[i];
                switch (ch)
                {
                    case '[':
                    case '(':
                    case '{': a.Push(ch); break;
                    case ')':
                    case '}':
                    case ']':
                        if (a.Count == 0)
                            return false;
                        if (pairmap[a.Peek()] == ch)
                        {
                            a.Pop();
                        }
                        else
                        {
                            return false;
                        }
                        break;
                }
            }
            return a.Count == 0;
        }
        #endregion

        #region   Solution 26
        public int Solution_0026(int[] nums)
        {
            int offset = 0;
            for (int i = 0; ;)
            {
                offset++;
                if (i + 1 >= nums.Length)
                    break;
                if (i + 1 < nums.Length && nums[i + 1] == nums[i])
                {
                    while (i + 1 < nums.Length && nums[i + 1] == nums[i])
                    {
                        i++;
                    }
                }
                i++;
                if (i >= nums.Length)
                    break;
                nums[offset] = nums[i];
            }

            return offset;
        }
        #endregion

        #region   Solution 3147
        public int Solution_3147(int[] energy, int k)
        {
            // dp[i] = max (e[i],e[i] + e[i-k])
            int max = int.MinValue;
            int[] dp = new int[energy.Length];
            for (int i = 0; i < energy.Length; i++)
            {
                dp[i] = energy[i];
                if (i - k >= 0)
                {
                    dp[i] = Math.Max(dp[i], dp[i - k] + energy[i]);
                }
            }

            for (int j = energy.Length - 1; j >= energy.Length - k; j--)
            {
                max = Math.Max(max, dp[j]);
            }

            return max;
        }
        #endregion

        #region   Solution 58
        public int Solution_0058(string s)
        {
            int max = 0;
            bool flag = true;
            for (int i = 0; i < s.Length; i++)
            {
                if (s[i] == ' ')
                {
                    flag = true;
                }
                else
                {
                    if (flag)
                    {
                        flag = false;
                        max = 0;
                    }
                    max++;
                }
            }
            return max;
        }
        #endregion

        #region   Solution 69
        public int Solution_0069(int x)
        {
            if (x == 0)
                return 0;

            int s = x;
            double sqrts(double x)
            {
                double res = (x + s / x) / 2;
                if (res == x)
                {
                    return x;
                }
                else
                {
                    return sqrts(res);
                }
            }

            return ((int)(sqrts(x)));
        }
        #endregion

        #region   Solution 101
        public bool Solution_0101(TreeNode root)
        {
            Queue<TreeNode> layer = new Queue<TreeNode>();
            layer.Enqueue(root);

            List<TreeNode> child = new List<TreeNode>();
            while (layer.Count > 0)
            {
                child.Clear();
                bool allnull = true;
                while (layer.Count > 0)
                {
                    var node = layer.Dequeue();
                    if (node != null)
                    {
                        child.Add(node.left);
                        child.Add(node.right);
                        allnull = false;
                    }
                    else
                    {
                        child.Add(null);
                        child.Add(null);
                    }
                }

                if (allnull)
                    break;

                for (int i = 0; i < child.Count - 1 - i; i++)
                {
                    if (child[i] == null && child[child.Count - 1 - i] == null)
                    {
                        continue;
                    }
                    if (child[i] != null && child[child.Count - 1 - i] != null &&
                        child[i].val == child[child.Count - 1 - i].val)
                    {
                        continue;
                    }

                    return false;
                }

                for (int i = 0; i < child.Count; i++)
                {
                    layer.Enqueue(child[i]);
                }
            }


            return true;
        }
        #endregion

        #region   Solution 23
        public ListNode Solution_0023(ListNode[] lists)
        {
            if (lists.Length == 0)
                return null;

            var heap = new Heap<ListNode>(new ListNodeComparer(), lists);
            if (heap.Count == 0)
                return null;

            ListNode newhead = new ListNode();
            ListNode current = newhead;
            while (heap.Count > 0 && heap.Peek() != null)
            {
                current.val = heap.Peek().val;
                if (heap.Peek().next != null)
                {
                    heap.Pop(heap.Peek().next);
                }
                else
                {
                    heap.RemoveTop();
                }

                if (heap.Count > 0)
                {
                    current.next = new ListNode();
                    current = current.next;
                }
            }

            return newhead;
        }
        #endregion

        #region   Solution 191
        public int Solution_0191(int n)
        {
            if (n == 0)
                return 0;
            return (n & 0x1) + Solution_0191(n >> 1);
        }
        #endregion

        #region   Solution 215
        private int QuickPick(int[] nums, int start, int end, int k)
        {
            if (start >= end)
                return nums[end];

            var pk = nums[start];
            var tempstart = start;
            var tempend = end;

            int same = 0;
            while (tempstart < tempend)
            {
                if (nums[tempstart] > pk)
                {
                    tempstart++;
                    continue;
                }

                if (nums[tempend] <= pk)
                {
                    if (nums[tempend] == pk)
                    {
                        same++;
                    }
                    tempend--;
                    continue;
                }

                var c = nums[tempend];
                nums[tempend] = nums[tempstart];
                nums[tempstart] = c;
            }

            if (tempstart > k)
            {
                return QuickPick(nums, start, tempstart - 1, k);
            }
            else if (tempstart < k)
            {
                if (tempstart + same < k)
                {
                    return pk;
                }
                return QuickPick(nums, tempstart == start ? (tempstart + 1) : tempstart, end, k);
            }
            else
            {
                return pk;
            }
        }

        public int Solution_0215(int[] nums, int k)
        {
            if (nums.Length == 1)
                return nums[0];

            int pk = QuickPick(nums, 0, nums.Length - 1, k - 1);
            return pk;
        }
        #endregion

        #region   Solution 3186
        public struct PowerInfo
        {
            public long power;
            public int count;
            public long sum;
        }

        public long Solution_3186(int[] power)
        {
            long max = 0;
            Dictionary<long, int> countdic = new Dictionary<long, int>();
            for (int i = 0; i < power.Length; i++)
            {
                if (!countdic.ContainsKey(power[i]))
                {
                    countdic[power[i]] = 0;
                }
                countdic[power[i]] += 1;
            }

            var kvs = countdic.OrderBy(kv => kv.Key).ToList();
            PowerInfo[] dp = new PowerInfo[countdic.Count];
            for (int i = 0; i < countdic.Count; i++)
            {
                var kv = kvs[i];
                dp[i] = new PowerInfo { power = kv.Key, count = kv.Value, sum = kv.Key * kv.Value };
                if (i - 1 >= 0)
                {
                    if (kv.Key - dp[i - 1].power > 2)
                    {
                        dp[i].sum += dp[i - 1].sum;
                    }
                    else
                    {
                        dp[i].sum = Math.Max(dp[i].sum, dp[i - 1].sum);
                    }
                }

                if (i - 2 >= 0)
                {
                    if (kv.Key - dp[i - 2].power > 2)
                    {
                        dp[i].sum = Math.Max(dp[i].sum, kv.Key * kv.Value + dp[i - 2].sum);
                    }
                    else
                    {
                        dp[i].sum = Math.Max(dp[i].sum, dp[i - 2].sum);
                    }
                }

                if (i - 3 >= 0 && kv.Key - dp[i - 3].power > 2)
                {
                    dp[i].sum = Math.Max(dp[i].sum, kv.Key * kv.Value + dp[i - 3].sum);
                }

                max = Math.Max(max, dp[i].sum);
            }

            return max;
        }
        #endregion

        #region   Solution 343
        public int Solution_0343(int n)
        {
            int[] dp = new int[n + 4];
            dp[0] = 0;
            dp[1] = 1;
            dp[2] = 1;
            dp[3] = 2;
            // n >= 5 可分
            for (int i = 4; i <= n; i++)
            {
                //var k = i / 4;
                //var last = i % 4;
                dp[i] = Math.Max(dp[i - 3] * 3, (i / 2) * (i - (i / 2)));
            }

            return dp[n];
        }
        #endregion

        #region   Solution 112
        private bool LeftTraversal(TreeNode root, int targetSum, int aim)
        {
            if (root.right != null || root.left != null)
            {
                if (root.left != null)
                {
                    if (LeftTraversal(root.left, targetSum + root.val, aim))
                    {
                        return true;
                    }
                }
                if (root.right != null)
                {
                    if (LeftTraversal(root.right, targetSum + root.val, aim))
                    {
                        return true;
                    }
                }
                return false;
            }
            else
            {
                if (targetSum + root.val == aim)
                    return true;

                return false;
            }
        }

        public bool Solution_0112(TreeNode root, int targetSum)
        {
            if (root == null)
                return false;
            return LeftTraversal(root, 0, targetSum);
        }
        #endregion

        #region   Solution 100
        public bool Solution_0100(TreeNode p, TreeNode q)
        {
            if (p == null && q != null || p != null && q == null)
                return false;

            if (p == null && q == null)
                return true;

            if (p.val != q.val)
                return false;

            return Solution_0100(p.left, q.left) && Solution_0100(p.right, q.right);
        }
        #endregion

        #region   Solution 108
        public TreeNode Solution_0108(int[] nums)
        {
            TreeNode root = null;
            for (int i = 0; i < nums.Length; i++)
            {
                var num = nums[i];
                if (root == null)
                {
                    root = new TreeNode(num);
                }
                else
                {

                }
            }
            return default;
        }
        #endregion

        #region   Solution 3539
        static double Factorial(int num)
        {
            if (num <= 1)
            {
                return 1;
            }
            return num * Factorial(num - 1);
        }

        static double Factorial(int[] nums, int i)
        {
            if (i == 0)
            {
                return nums[i];
            }
            return nums[i] * Factorial(nums, i - 1);
        }

        public static void Solution_3539(int[] nums, int start, int k, long currentsum, List<long> sum)
        {
            if (start >= nums.Length)
                return;

            currentsum *= nums[start];
            if (k == 0)
            {
                sum.Add(currentsum);
                return;
            }
            while (start < nums.Length)
            {
                Solution_3539(nums, start + 1, k - 1, currentsum, sum);
                start++;
            }
        }

        public int Solution_3539(int m, int k, int[] nums)
        {
            // DP [][][]
            // 
            // DFS 
            if (m == nums.Length)
                return (int)(Factorial(m) * Factorial(nums, nums.Length - 1) % (1000000000 + 7));

            if (m == k)
            {
                double factor = Factorial(m);
                double sum = 0;
                List<long> sums = new List<long>();
                for (int i = 0; i <= nums.Length - m; i++)
                {
                    sums.Clear();
                    Solution_3539(nums, i, k - 1, 1, sums);
                    sum += sums.Sum();
                }
                return (int)((sum * factor) % (1000000000 + 7));
            }
            else
            {

            }

            return default;

        }
        #endregion

        #region   Solution 190
        private int ReverseBits(int n, int offset)
        {
            if (n == 0)
                return 0;
            return (int)((n & 0x80000000) >> offset) + ReverseBits(n << 1, offset - 1);
        }

        public int Solution_0190(int n)
        {
            return ReverseBits(n, 31);
        }
        #endregion

        #region   Solution 1763
        public string IsNiceSubstring(string s, int length, HashSet<char> cset)
        {
            if (length == 0)
                return "";

            for (int i = 0; i <= s.Length - length; i++)
            {
                var sub = s.Substring(i, length);
                cset.Clear();
                cset.UnionWith(sub);
                bool flag = false;
                if (cset.Count % 2 == 0)
                {
                    flag = true;
                    foreach (var c in cset)
                    {
                        if (c < 'a')
                        {
                            if (!cset.Contains((char)(c + ('a' - 'A'))))
                            {
                                flag = false;
                                break;
                            }
                        }
                        else
                        {
                            if (!cset.Contains((char)(c + ('A' - 'a'))))
                            {
                                flag = false;
                                break;
                            }
                        }
                    }
                }
                if (flag)
                    return sub;
            }

            return IsNiceSubstring(s, length - 1, cset);
        }

        public string Solution_1763(string s)
        {
            HashSet<char> sd = new HashSet<char>();
            return IsNiceSubstring(s, s.Length, sd);
        }
        #endregion

        #region   Solution 159
        public int[] Solution_0159(int[] stock, int cnt)
        {
            if (stock == null || cnt == 0)
                return new int[0];

            Dictionary<int, int> list = new Dictionary<int, int>();
            for (int i = 0; i < stock.Length; i++)
            {
                if (!list.ContainsKey(stock[i]))
                {
                    list[stock[i]] = 0;
                }
                list[stock[i]]++;
            }

            List<int> finaldata = new List<int>();
            var odlist = list.OrderBy(kv => kv.Key).ToList();
            foreach (var item in odlist)
            {
                for (int i = 0; i < item.Value; i++)
                {
                    finaldata.Add(item.Key);
                    if (finaldata.Count == cnt)
                        return finaldata.ToArray();
                }
            }

            return default;
        }
        #endregion

        #region   Solution 2273
        public IList<string> Solution_2273(string[] words)
        {
            List<string> newwords = new List<string>();
            List<char> key = null;
            for (int i = 0; i < words.Length; i++)
            {
                if (key == null)
                {
                    key = words[i].ToList();
                    key.Sort();
                    newwords.Add(words[i]);
                }
                else
                {
                    var current = words[i].ToList();
                    current.Sort();
                    if (current.SequenceEqual(key))
                    {
                        continue;
                    }
                    else
                    {
                        key = current;
                        newwords.Add(words[i]);
                    }
                }
            }

            return newwords;
        }
        #endregion

        #region   Solution 347
        public int[] Solution_0347(int[] nums, int k)
        {
            if (nums.Length == 1)
                return nums;

            return default;
        }
        #endregion

        #region   Solution 372
        public int Solution_0372(int a, int[] b)
        {
            return default;
        }
        #endregion

        #region   Solution 3349
        public bool Solution_3349(IList<int> nums, int k)
        {
            if (k == 1)
                return true;

            int start = 0, end = 0;
            List<(int e, int s)> decse = new List<(int e, int s)>();
            for (int i = 0; i < nums.Count; i++)
            {
                if (nums[i] > nums[end])
                {
                    if (i - start > 1)
                    {
                        decse.Add((start, end));
                    }
                    start = i;
                }
                end = i;
            }

            if (start != nums.Count - 1)
            {
                decse.Add((start, nums.Count - 1));
            }

            if (decse.Count > 0)
            {
                int prestart = 0;
                int prelen = 0;
                for (int i = 0; i < decse.Count; i++)
                {
                    if (i - 1 >= 0 && decse[i - 1].s - decse[i - 1].e == 1)
                    {
                        if (prelen >= k && decse[i].e - prestart + 1 >= k)
                        {
                            return true;
                        }
                    }
                    prelen = decse[i].e - prestart + 1;
                    if (prelen >= 2 * k)
                        return true;
                    prestart = decse[i].s;
                }

                if (decse[decse.Count - 1].e != nums.Count - 1)
                {
                    if (nums.Count - prestart >= 2 * k)
                        return true;
                    if (decse[decse.Count - 1].s - decse[decse.Count - 1].e == 1)
                    {
                        if (prelen >= k && nums.Count - prestart >= k)
                        {
                            return true;
                        }
                    }
                }
            }
            else
            {
                return nums.Count >= 2 * k;
            }

            return false;
        }
        #endregion

        #region   Solution 105
        private TreeNode BuildTree(int[] preorder, ref int current, int[] inorder, int start, int end)
        {
            if (current >= preorder.Length)
                return null;
            TreeNode root = new TreeNode { val = preorder[current] };

            int i = start;
            for (; i < end; i++)
            {
                if (inorder[i] == root.val)
                {
                    break;
                }
            }

            if (i - start > 0)
            {
                current++;
                root.left = BuildTree(preorder, ref current, inorder, start, i - 1);
            }
            if (end - i > 0)
            {
                current++;
                root.right = BuildTree(preorder, ref current, inorder, i + 1, end);
            }
            return root;
        }

        public TreeNode Solution_0105(int[] preorder, int[] inorder)
        {
            if (preorder == null || inorder == null || preorder.Length == 0 || inorder.Length == 0)
                return null;
            int current = 0;
            return BuildTree(preorder, ref current, inorder, 0, inorder.Length);
        }
        #endregion

        #region   Solution 2598
        public int Solution_2598(int[] nums, int value)
        {
            var numsdic = new int[value];

            foreach (var item in nums)
            {
                var num = (item % value);
                if (num < 0)
                {
                    num += value;
                }
                numsdic[num]++;
            }

            int min = int.MaxValue;
            int minnum = 0;
            for (int i = 0; i < value; i++)
            {
                if (numsdic[i] < min)
                {
                    min = numsdic[i];
                    minnum = i;
                }
            }

            return value * min + minnum;
        }
        #endregion

        #region   Solution 3350
        public int Solution_3350(IList<int> nums)
        {
            List<(int e, int s)> descs = new List<(int e, int s)>();
            int start = 0, end = 0;
            for (int i = 0; i < nums.Count; i++)
            {
                if (nums[i] > nums[end])
                {
                    if (start != end)
                    {
                        descs.Add((start, end));
                    }
                    start = i;
                }
                end = i;
            }
            if (start != end)
            {
                descs.Add((start, end));
            }

            if (descs.Count == 0)
            {
                return nums.Count / 2;
            }

            int maxlen = 1;
            for (int i = 0; i < descs.Count; i++)
            {
                if (descs[i].s - descs[i].e == 1)
                {
                    int length = 0;
                    if (i - 1 >= 0)
                    {
                        length = descs[i].e - descs[i - 1].s + 1;
                    }
                    else
                    {
                        length = descs[i].e - 0 + 1;
                    }
                    if (i + 1 < descs.Count)
                    {
                        length = Math.Min(length, descs[i + 1].e - descs[i].s + 1);
                    }
                    else
                    {
                        length = Math.Min(length, nums.Count - descs[i].s);
                    }
                    maxlen = Math.Max(maxlen, length);
                }

                if (i - 1 >= 0)
                {
                    maxlen = Math.Max(maxlen, (descs[i].e - descs[i - 1].s + 1) / 2);
                }
                else
                {
                    maxlen = Math.Max(maxlen, (descs[i].e + 1) / 2);
                }
            }

            if (descs[descs.Count - 1].s != nums.Count - 1)
            {
                maxlen = Math.Max(maxlen, (nums.Count - descs[descs.Count - 1].s) / 2);
            }

            return maxlen;
        }
        #endregion

        #region   Solution 7
        public int Solution_0007(int x)
        {
            int[] xmax = new int[] { 2, 1, 4, 7, 4, 8, 3, 6, 4, 7 };
            int[] xmin = new int[] { 2, 1, 4, 7, 4, 8, 3, 6, 4, 8 };

            int digit = 0;
            int zeroend = 0;
            bool flag = true;
            bool positive = x >= 0;

            List<int> digits = new List<int>();
            x = positive ? x : -x;
            while (x > 0)
            {
                var current = x % 10;
                if (current == 0 && flag)
                {
                    zeroend++;
                }
                else
                {
                    flag = false;
                }
                if (!flag)
                {
                    digits.Add(current);
                }
                x = x / 10;
                digit++;
            }

            int rev = 0;
            if (digits.Count == 10)
            {
                bool flag1 = true;
                for (int i = 0; i < digits.Count; i++)
                {
                    if (positive)
                    {
                        if (digits[i] > xmax[i] && flag1)
                        {
                            return 0;
                        }
                        else if (digits[i] < xmax[i])
                        {
                            flag1 = false;
                        }
                    }
                    else
                    {
                        if (digits[i] > xmin[i] && flag1)
                        {
                            return 0;
                        }
                        else if (digits[i] < xmin[i])
                        {
                            flag1 = false;
                        }
                    }
                    rev += digits[i] * (int)Math.Pow(10, digits.Count - (i + 1));
                }
            }
            else
            {
                for (int i = 0; i < digits.Count; i++)
                {
                    rev += digits[i] * (int)Math.Pow(10, digits.Count - (i + 1));
                }
            }

            return rev * (positive ? 1 : -1);
        }
        #endregion

        #region   Solution 1135
        public struct CostI
        {
            public CostI(int x, int y, int cost)
            {
                X = x;
                Y = y;
                COST = cost;
            }

            public int X { get; set; }
            public int Y { get; set; }
            public int COST { get; set; }
        }

        public int Solution_1135(int n, int[][] connections)
        {
            if (connections.Length < n - 1)
                return -1;

            List<CostI> costs = new List<CostI>();
            for (int i = 0; i < connections.Length; i++)
            {
                int xi = Math.Min(connections[i][0], connections[i][1]);
                int yi = Math.Max(connections[i][0], connections[i][1]);
                int costi = connections[i][2];

                costs.Add(new CostI(xi, yi, costi));
            }

            costs.Sort((a, b) => a.COST - b.COST);
            int sum = 0;
            UnionFind uf = new UnionFind(n);
            int edge = 0;
            for (int i = 0; i <= costs.Count; i++)
            {
                if (edge == n - 1)
                    return sum;
                if (i == costs.Count)
                    continue;

                var xp = uf.Find(costs[i].X);
                var yp = uf.Find(costs[i].Y);
                if (xp == yp)
                {
                    continue;
                }
                sum += costs[i].COST;
                uf.Union(costs[i].X, costs[i].Y);
                edge++;
            }

            return -1;
        }
        #endregion

        #region   Solution 10
        public bool Solution_10(string s, int start, List<string> elements, bool endwithstart)
        {
            if (elements.Count == 0)
                return start == s.Length;

            var element = elements.First();
            int index = 0;
            int i = start;
            while (i < s.Length && index < element.Length)
            {
                if (index == element.Length - 1 && elements.Count > 1)
                {
                    elements.RemoveAt(0);
                    if (Solution_10(s, i, elements.ToList(), endwithstart))
                    {
                        return true;
                    }
                    elements.Insert(0, element);
                }

                if (element[index] == '.' || s[i] == element[index])
                {
                    index++;
                    i++;
                }
                else if (s[i] != element[index])
                {
                    return false;
                }
            }

            if (i == s.Length && index < element.Length - 1)
            {
                return false;
            }

            elements.RemoveAt(0);
            var last = element.Last();
            if (last == '.')
            {
                if (elements.Count == 0)
                    return (index >= element.Length - 1 && endwithstart) || (i == s.Length && index == element.Length);

                bool flag = false;
                while (i <= s.Length)
                {
                    flag |= Solution_10(s, i, elements.ToList(), endwithstart);
                    if (flag)
                        return true;
                    i++;
                }
                return flag;
            }
            else
            {
                if (elements.Count == 0)
                {
                    if (endwithstart)
                    {
                        if (index < element.Length - 1)
                            return false;
                        while (i < s.Length)
                        {
                            if (s[i] != last)
                            {
                                return false;
                            }
                            i++;
                        }
                        return true;
                    }
                    else
                    {
                        return i == s.Length && index == element.Length;
                    }
                }

                bool flag = false;
                while (i <= s.Length)
                {
                    if (i < s.Length && s[i] != last)
                    {
                        flag |= Solution_10(s, i, elements.ToList(), endwithstart);
                        break;
                    }
                    flag |= Solution_10(s, i, elements.ToList(), endwithstart);
                    if (flag)
                        return true;
                    i++;
                }
                return flag;
            }
        }

        public bool Solution_10_DP(string s, string p)
        {
            // dp[i,j] = 
            bool[,] dp = new bool[s.Length + 1, p.Length + 1];

            dp[0, 0] = true;
            for (int i = 1; i < s.Length + 1; i++)
            {
                dp[i, 0] = false;
            }

            for (int j = 1; j < p.Length + 1; j++)
            {
                if (p[j - 1] == '*')
                {
                    dp[0, j] = dp[0, j - 1] || dp[0, j - 2];
                }
            }

            for (int j = 1; j < p.Length + 1; j++)
            {
                switch (p[j - 1])
                {
                    case '.':
                        // 当前字符为 '.' 表示匹配任意字符 只要之前能匹配 那当前就能匹配
                        for (int k = 1; k < s.Length + 1; k++)
                        {
                            if (dp[k - 1, j - 1])
                            {
                                dp[k, j] = true;
                            }
                        }
                        break;
                    case '*':
                        // 当前字符为 '*' 只要之前能匹配 当前便能匹配
                        for (int k = 1; k < s.Length + 1; k++)
                        {
                            if (dp[k, j - 1] || (j >= 2 && dp[k, j - 2]))
                            {
                                dp[k, j] = true;
                            }
                            else if (j >= 2 && (s[k - 1] == p[j - 2] || p[j - 2] == '.') && dp[k - 1, j])
                            {
                                dp[k, j] = true;
                            }
                        }
                        break;
                    default:
                        // 当前字符为特定字符 只要之前能匹配 且当前字符等于特定字符 那当前就能匹配
                        for (int k = 1; k < s.Length + 1; k++)
                        {
                            if (dp[k - 1, j - 1] && s[k - 1] == p[j - 1])
                            {
                                dp[k, j] = true;
                            }
                        }
                        break;
                }
            }

            return dp[s.Length, p.Length];
        }

        public bool Solution_0010(string s, string p)
        {
            return Solution_10_DP(s, p);
        }
        #endregion

        #region   Solution 3003

        public class Partition3003
        {
            public int Start { get; set; }
            public int End { get; set; }
            public int[] Exist { get; } = Enumerable.Repeat(-1, 26).ToArray();
            public int DiffCount { get; set; }
            public int Remain { get; set; }
            public List<int> CoverNumber { get; } = new List<int>();
            public List<int> Positions { get; } = new List<int>();
            public bool FullCover { get; set; }
            public int LeftMax { get; set; } = -1;
            public int RightMin { get; set; } = -1;
            public int RightNewEnd { get; set; }
            public int Increase { get; set; }
            public bool HasTwoPart => RightMin - LeftMax > 1;
        }

        public int CheckPartition(string s, List<Partition3003> parts, int index, HashSet<int> added)
        {
            if (index >= parts.Count)
                return added.Count > 0 ? 1 : 0;

            var current = parts[index];
            // 前面后移的元素能被吸收
            if (current.FullCover)
                return 0;

            added.ExceptWith(current.CoverNumber);
            // 前面后移的元素能被吸收
            if (added.Count == 0)
                return 0;

            // 最后一个分部不满 可以吸收
            if (current.Remain > 0)
            {
                return current.Remain >= added.Count ? 0 : 1;
            }

            var newend = current.Positions[current.Positions.Count - added.Count];
            added.Clear();
            for (int i = newend; i <= current.End; i++)
            {
                added.Add(s[i] - 'a');
            }

            return CheckPartition(s, parts, index + 1, added);
        }

        public int Solution_3003(string s, int k)
        {
            if (k == 26)
                return 1;
            // 子串中 最小划分 最多的 且余数

            List<Partition3003> partitions = new List<Partition3003>();
            Partition3003 current = new Partition3003();
            HashSet<int> tempset = new HashSet<int>();
            List<int> hastwopart = new List<int>();
            int lastfullcover = 0;
            bool overtwo = false;
            for (int i = 0; i <= s.Length; i++)
            {
                if (i == s.Length || current.Exist[s[i] - 'a'] < 0)
                {
                    if (current.DiffCount == k || i == s.Length)
                    {
                        tempset.Clear();
                        if (partitions.Count > 0)
                        {
                            for (int j = 0; j < 26; j++)
                            {
                                if (current.Exist[j] >= 0 && partitions.Last().Exist[j] >= 0)
                                {
                                    current.CoverNumber.Add(j);
                                }
                            }
                            if (current.CoverNumber.Count == k)
                            {
                                current.FullCover = true;
                                lastfullcover = partitions.Count;
                            }
                        }
                        current.End = i - 1;
                        if (current.LeftMax < 0)
                        {
                            current.LeftMax = i - 1;
                        }

                        for (int kk = i - 1; kk >= current.LeftMax && kk >= 0; kk--)
                        {
                            tempset.Add(s[kk] - 'a');
                            if (tempset.Count == 2)
                            {
                                current.RightNewEnd = kk +1;
                            }
                            if (tempset.Count == k)
                            {
                                current.RightMin = kk;
                                break;
                            }
                        }
                        if (current.RightMin < 0)
                        {
                            current.RightMin = current.Start;
                        }
                        if (current.HasTwoPart)
                        {
                            hastwopart.Add(partitions.Count);
                            tempset.Clear();
                            int times = 0;
                            for (int u = current.Start; u <= current.End; u++)
                            {
                                tempset.Add(s[u]);
                                if (tempset.Count == k)
                                {
                                    tempset.Clear();
                                    times++;
                                }
                                if (times > 2)
                                {
                                    current.Increase = 1;
                                    break;
                                }
                            }
                        }
                        current.Remain = k - current.DiffCount;
                        if (current.End - current.Start >= 1)
                        {
                            overtwo = true;
                        }
                        partitions.Add(current);
                        current = new Partition3003();
                        current.Start = i;
                    }

                    if (i < s.Length)
                    {
                        current.Exist[s[i] - 'a'] = i;
                        current.Positions.Add(i);
                        current.DiffCount++;
                        if (current.DiffCount == k)
                        {
                            current.LeftMax = i;
                        }
                    }
                }
            }

            if (partitions.Count == 1 && partitions[0].DiffCount < k)
            {
                return 1;
            }

            HashSet<int> toadd = new HashSet<int>();
            int count = partitions.Count;
            if (hastwopart.Count > 0)
            {
                if (k == 1)
                    return partitions.Count + 2;

                count += 1;
                foreach (var pos in hastwopart)
                {
                    toadd.Clear();
                    var currentpart = partitions[pos];
                    for (int j = currentpart.RightNewEnd; j <= currentpart.End; j++)
                    {
                        toadd.Add(s[j] - 'a');
                    }
                    int temp = currentpart.Increase;
                    if (CheckPartition(s, partitions, pos + 1, toadd) > 0)
                    {
                        temp = 1;
                    }
                    count = Math.Max(count ,count + temp);
                }
            }
            else
            {
                if (k == 1)
                    return overtwo ? partitions.Count + 1 : partitions.Count;

                for (int i = lastfullcover; i < partitions.Count; i++)
                {
                    var currentpart = partitions[i];
                    if (currentpart.End - currentpart.Start + 1 <= k)
                        continue;

                    if (i == partitions.Count - 1)
                    {
                        return partitions[i].Remain > 0 ? count : count + 1;
                    }

                    toadd.Clear();
                    if (currentpart.Positions[currentpart.Positions.Count - 1] != currentpart.Positions.Count - 1 + currentpart.Start)
                    {
                        bool flag = true;
                        if (currentpart.Positions[currentpart.Positions.Count - 2] == currentpart.Positions.Count - 2 + currentpart.Start)
                        {
                            for (int j = currentpart.Positions[currentpart.Positions.Count - 2] + 2; j <= currentpart.End; j++)
                            {
                                toadd.Add(s[j] - 'a');
                            }
                        }
                        else
                        {
                            for (int j = currentpart.Positions[currentpart.Positions.Count - 2] + 1; j <= currentpart.End; j++)
                            {
                                if (currentpart.Exist[s[j] - 'a'] >= 0 && s[j] != s[currentpart.Positions.Last()])
                                {
                                    if (flag)
                                        continue;
                                }
                                else
                                {
                                    flag = false;
                                }
                                toadd.Add(s[j] - 'a');
                            }
                        }
                    }
                    else
                    {
                        for (int u = 0; u < partitions[i + 1].Exist.Length; u++)
                        {
                            if (partitions[i + 1].Exist[u] == -1 && !toadd.Contains(u) && currentpart.Exist[u] == -1)
                            {
                                toadd.Add(u);
                                break;
                            }
                        }
                        if (toadd.Count == 0)
                        {
                            for (int j = currentpart.Positions.Last() + 1; j <= currentpart.End; j++)
                            {
                                toadd.Add(s[j] - 'a');
                            }
                        }
                        else
                        {
                            for (int j = currentpart.Positions.Last() + 2; j <= currentpart.End; j++)
                            {
                                toadd.Add(s[j] - 'a');
                            }
                        }
                    }

                    if (toadd.Count == k)
                        return count + 1;

                    if (CheckPartition(s, partitions, i + 1, toadd) > 0)
                    {
                        count += 1;
                        break;
                    }
                }
            }

            return count;
        }

        #endregion

        #region   Solution 125
        public bool Solution_125(string s)
        {
            var strnew = System.Text.RegularExpressions.Regex.Replace(s.ToLower(), "[^0-9A-Za-z]", "");
            int start = 0, end = strnew.Length - 1;
            while (start < end)
            {
                if (strnew[start] != strnew[end])
                {
                    return false;
                }
                start++;
                end--;
            }

            return true;
        }
        #endregion

        #region   Solution 3397
        public int Solution_3397(int[] nums, int k)
        {
            List<(long low, long high)> numsrange = new List<(long low, long high)>();
            long offset = k > 0 ? k : -k;
            for (int i = 0; i < nums.Length; i++)
            {
                numsrange.Add((nums[i] - offset, nums[i] + offset));
            }

            numsrange.Sort((a, b) => (int)(a.low - b.low));
            long current = numsrange[0].low;
            int sum = 1;
            for (int i = 1; i < numsrange.Count; i++)
            {
                if (numsrange[i].high > current)
                {
                    sum += 1;
                    if (current < numsrange[i].low)
                    {
                        current = numsrange[i].low;
                    }
                    else
                    {
                        current++;
                    }
                }
            }

            return sum;
        }
        #endregion
    }
}
