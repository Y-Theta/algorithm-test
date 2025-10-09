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
    }
}
