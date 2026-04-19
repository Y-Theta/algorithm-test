from Common import SegmentTreeNode
from Common import ListNode
from typing import Optional, List, Dict, Counter, Tuple, Set

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
    
class ZigzagIterator:

    def __init__(self, v1: List[int], v2: List[int]):
        self.v1 = v1
        self.v2 = v2
        self.v1index = 0
        self.v2index = 0
        self.cp = 0

    def next(self) -> int:
        num= 0
        if self.cp == 0:
            num = self.getv1()
            if num == None:
                num = self.getv2()
        else:
            num = self.getv2()
            if num == None:
                num = self.getv1()
        self.cp += 1
        self.cp %= 2
        return num

    def getv1(self) -> Optional[int]:
        if self.v1index < len(self.v1):
            num = self.v1[self.v1index]
            self.v1index += 1
            return num
        return None
    
    def getv2(self) -> Optional[int]:
        if self.v2index < len(self.v2):
            num = self.v2[self.v2index]
            self.v2index += 1
            return num
        return None

    def hasNext(self) -> bool:
        return self.v1index < len(self.v1) or self.v2index < len(self.v2)