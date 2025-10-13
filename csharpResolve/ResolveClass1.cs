using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace main.csharpResolve
{
    internal class ResolveClass1
    {
        public long MinTime_3494(int[] skill, int[] mana)
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

        public bool IsValid_0020(string s)
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

        public int RemoveDuplicates_0026(int[] nums)
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

        public int MaximumEnergy_3147(int[] energy, int k)
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

        public int LengthOfLastWord_0058(string s)
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

        public int MySqrt_0069(int x)
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

        public bool IsSymmetric_0101(TreeNode root)
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

        public ListNode MergeKLists_0023(ListNode[] lists)
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

        public int HammingWeight_0191(int n)
        {
            if (n == 0)
                return 0;
            return (n & 0x1) + HammingWeight_0191(n >> 1);
        }

        private int quickPick(int[] nums, int start, int end, int k)
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
                return quickPick(nums, start, tempstart - 1, k);
            }
            else if (tempstart < k)
            {
                if (tempstart + same < k)
                {
                    return pk;
                }
                return quickPick(nums, tempstart == start ? (tempstart + 1) : tempstart, end, k);
            }
            else
            {
                return pk;
            }
        }

        public int FindKthLargest_0215(int[] nums, int k)
        {
            if (nums.Length == 1)
                return nums[0];

            int pk = quickPick(nums, 0, nums.Length - 1, k - 1);
            return pk;
        }

        #region   
        public struct PowerInfo
        {
            public long power;
            public int count;
            public long sum;
        }

        public long MaximumTotalDamage_3186(int[] power)
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

        public int IntegerBreak_0343(int n)
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

        public bool HasPathSum_0112(TreeNode root, int targetSum)
        {
            if (root == null)
                return false;
            return LeftTraversal(root, 0, targetSum);
        }

        public bool IsSameTree_0100(TreeNode p, TreeNode q)
        {
            if (p == null && q != null || p != null && q == null)
                return false;

            if (p == null && q == null)
                return true;

            if (p.val != q.val)
                return false;

            return IsSameTree_0100(p.left, q.left) && IsSameTree_0100(p.right, q.right);
        }

        public TreeNode SortedArrayToBST_0108(int[] nums)
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

        public static void DFS_3539(int[] nums, int start, int k, long currentsum, List<long> sum)
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
                DFS_3539(nums, start + 1, k - 1, currentsum, sum);
                start++;
            }
        }

        public int MagicalSum_3539(int m, int k, int[] nums)
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
                    DFS_3539(nums, i, k - 1, 1, sums);
                    sum += sums.Sum();
                }
                return (int)((sum * factor) % (1000000000 + 7));
            }
            else
            {

            }

            return default;

        }

        private int ReverseBits(int n, int offset)
        {
            if (n == 0)
                return 0;
            return (int)((n & 0x80000000) >> offset) + ReverseBits(n << 1, offset - 1);
        }

        public int ReverseBits_0190(int n)
        {
            return ReverseBits(n, 31);
        }

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

        public string LongestNiceSubstring_1763(string s)
        {
            HashSet<char> sd = new HashSet<char>();
            return IsNiceSubstring(s, s.Length, sd);
        }

        public int[] InventoryManagement_0159(int[] stock, int cnt)
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

        public IList<string> RemoveAnagrams_2273(string[] words)
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

        public int[] TopKFrequent_0347(int[] nums, int k)
        {
            if (nums.Length == 1)
                return nums;

            return default;
        }

        public int GetPow(int x,int n)
        {
            int res = 1;
            while (n != 0)
            {
                if (n % 2 != 0)
                {
                    res = (int)((long)res * x /*% MOD*/);
                }
                x = (int)((long)x * x /*% MOD*/);
                n /= 2;
            }
            return res;
        }

        public int SuperPow_0372(int a, int[] b)
        {
            return default;
        }
    }
}
