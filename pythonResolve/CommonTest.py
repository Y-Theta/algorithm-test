from Common import SegmentTreeNode
from Common import ListNode

root: SegmentTreeNode = SegmentTreeNode(0, 0, SegmentTreeNode.minoperation)
numlist = [1, 3, 4, 5, 6, 6, 3, 1, 12, 3, 4, 5, 23, 2, 3, 2, 2, 1]
root.start = 0
root.end = len(numlist) - 1
SegmentTreeNode.build_segment_tree(numlist, root, 0, len(numlist) - 1)
None


class MRUQueue:

    def __init__(self, n: int):
        listnode = ListNode()
        self.head = listnode
        for i in range(n):
            listnode.val = i+1
            listnode.next = ListNode()
            current = listnode
            listnode = listnode.next
        self.tail = current
        current.next = None
        return

    def fetch(self, k: int) -> int:
        fore = None
        current = self.head
        for i in range(k-1):
            fore = current
            current = current.next
        
        if current == self.tail:
            return current.val
        
        if fore == None:
            self.head = current.next
        else:
            fore.next = current.next
        self.tail.next = current
        current.next = None
        self.tail = current
        return current.val