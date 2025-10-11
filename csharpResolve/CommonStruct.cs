using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace main.csharpResolve
{

    public class CommonUtils
    {
        public static int QuickPickK(IEnumerable<int> nums, int k)
        {
            return default;
        }
    }

    public class Heap<T> where T : class
    {
        private readonly List<T> __list = new List<T>();

        public int Count => __list.Count;

        private IComparer<T> _comparer;

        public Heap(IComparer<T> comparer)
        {
            _comparer = comparer;
        }

        public Heap(IComparer<T> comparer, T[] lists) : this( comparer)
        {
            foreach (var item in lists)
            {
                if (item != null)
                    __list.Add(item);
            }
            InitAlter();
        }

        public void Add(T node)
        {
            __list.Add(node);
        }

        public T Peek()
        {
            return __list[0];
        }

        public T Pop(T toadd = null)
        {
            var c = __list[0];
            if (toadd != null)
            {
                __list[0] = toadd;
            }
            else
            {
                __list[0] = __list[__list.Count - 1];
                __list.RemoveAt(__list.Count - 1);
            }
            if (__list.Count > 0)
            {
                Alter(0);
            }
            return c;
        }

        public void RemoveTop()
        {
            Pop();
        }

        private void InitAlter()
        {
            for (int i = __list.Count - 1; i > 0; i--)
            {
                var parent = (i - 1) / 2;
                if (parent >= 0 && _comparer.Compare(__list[parent], __list[i]) > 0)
                {
                    var c = __list[parent];
                    __list[parent] = __list[i];
                    __list[i] = c;
                    Alter(i);
                }
            }
        }

        private void Alter(int index)
        {
            if (index < 0 || index >= __list.Count)
                return;

            var leftchild = (index + 1) * 2 - 1;
            var rightchild = (index + 1) * 2;
            if (leftchild < __list.Count && _comparer.Compare(__list[index] , __list[leftchild]) > 0)
            {
                var c = __list[index];
                __list[index] = __list[leftchild];
                __list[leftchild] = c;
                Alter(leftchild);
            }

            if (rightchild < __list.Count && _comparer.Compare(__list[index], __list[rightchild]) > 0)
            {
                var c = __list[index];
                __list[index] = __list[rightchild];
                __list[rightchild] = c;
                Alter(rightchild);
            }

        }
    }

    public class ListNodeComparer : IComparer<ListNode>
    {
        public int Compare(ListNode x, ListNode y)
        {
            return x.val - y.val;
        }
    }

    public class ListNode
    {
        public int val;
        public ListNode next;
        public ListNode(int val = 0, ListNode next = null)
        {
            this.val = val;
            this.next = next;
        }

        public static ListNode ToLinkList(IEnumerable<int> items)
        {
            ListNode head = new ListNode();
            ListNode current = head;
            var count = items.Count();
            int index = 0;
            foreach (var item in items)
            {
                current.val = item;
                index++;
                if (index < count)
                {
                    current.next = new ListNode();
                    current = current.next;
                }
            }
            return head;
        }
    }

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
