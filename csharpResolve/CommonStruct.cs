using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace main.csharpResolve
{
    public class TreeNode
    {
        public int val;
        public TreeNode left;
        public TreeNode right;
        public TreeNode(int val = 0, TreeNode left = null, TreeNode right = null)
        {
            this.val = val;
            this.left = left;
            this.right = right;
        }

        public static void LeftTraversal(TreeNode root, List<int> list)
        {
            if (root.left != null)
            {
                LeftTraversal(root.left, list);
            }
            list.Add(root.val);
            if (root.right != null)
            {
                LeftTraversal(root.right, list);
            }
        }

        public static void RightTraversal(TreeNode root, List<int> list)
        {
            if (root.right != null)
            {
                RightTraversal(root.right, list);
            }
            list.Add(root.val);
            if (root.left != null)
            {
                RightTraversal(root.left, list);
            }
        }
    }
}
