using main.csharpResolve;

using System;
using System.Collections.Generic;

namespace leetcode
{
    public class Program
    {
        public static void Main(string[] args)
        {
            ResolveClass1 clas = new ResolveClass1();
            List< ListNode> nodes = new List< ListNode>();
            nodes.Add(ListNode.ToLinkList(new int[] { -6, -3, -1, 1, 2, 2, 2 }));
            nodes.Add(ListNode.ToLinkList(new int[] { -10, -8, -6, -2, 4 }));
            nodes.Add(ListNode.ToLinkList(new int[] { -2 }));
            nodes.Add(ListNode.ToLinkList(new int[] { -8, -4, -3, -3, -2, -1, 1, 2, 3 }));
            nodes.Add(ListNode.ToLinkList(new int[] { -8, -6, -5, -4, -2, -2, 2, 4 }));
            clas.HasPathSum_0112(new TreeNode
            {
                val = 1,
                left = new TreeNode
                {
                    val = 2,
             
                },
            },1);
        }
    }
}
