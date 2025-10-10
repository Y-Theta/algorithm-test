using main.csharpResolve;

using System;

namespace leetcode
{
    public class Program
    {
        public static void Main(string[] args)
        {
            ResolveClass1 clas = new ResolveClass1();
            TreeNode root = new TreeNode();
            root.val = 1;
            root.left = new TreeNode { left = new TreeNode() { val = 3 },right = new TreeNode { val = 4},val = 2 };
            root.right = new TreeNode { left = new TreeNode() { val = 4 },right = new TreeNode { val = 3 },val = 2 };
            clas.IsSymmetric_0101(root);
        }
    }
}
