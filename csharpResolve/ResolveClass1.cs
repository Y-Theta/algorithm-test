using System;
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
                lasttime[i] = lasttime[i-1] + (skill[i] * mana[0]);
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
                    lasttime[k] = lasttime[k-1] + (current * skill[k]);
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

            return offset ;
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
    }
}
