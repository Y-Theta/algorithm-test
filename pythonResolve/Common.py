from typing import Optional, List, Callable
from math import inf


class ListNode:
    def __init__(self, val: int = 0, next: Optional["ListNode"] = None):
        self.val = val
        self.next = next


class TreeNode:
    def __init__(
        self,
        val: int = 0,
        left: Optional["TreeNode"] = None,
        right: Optional["TreeNode"] = None,
    ):
        self.val = val
        self.left = left
        self.right = right


class SparseVector:
    def __init__(self, nums: List[int]):
        self._dict = dict()
        for i in range(len(nums)):
            if nums[i] > 0:
                self._dict[i] = nums[i]
        self._keyset = set(self._dict.keys())

    # Return the dotProduct of two sparse vectors
    def dotProduct(self, vec: "SparseVector") -> int:
        interct = self._keyset.intersection(vec._keyset)
        sum = 0
        for i in interct:
            sum += self._dict[i] * vec._dict[i]
        return sum


class SegmentTreeNode:
    minoperation:Callable[[int,int],int] = lambda a, b: min(a, b)
    maxoperation:Callable[[int,int],int] = lambda a, b: max(a, b)
    sumoperation:Callable[[int,int],int] = lambda a, b: a + b

    def __init__(self, start: int, end: int, dataoperation:Callable[[int,int],int]):
        self.start = start
        self.end = end
        self.left: Optional["SegmentTreeNode"] = None
        self.right: Optional["SegmentTreeNode"] = None
        self.value = inf  # 可以根据需要初始化为其他值或区间合并的结果
        self.dataoperation = dataoperation

    def __repr__(self):
        return (
            f"([{self.start}-{self.end}] {self.value}, L:{self.left}, R:{self.right})"
        )

    def build_segment_tree(
        arr: List[int], node: "SegmentTreeNode", start: int, end: int
    ):
        if start == end:
            node.value = arr[start]
            return node
        mid = (start + end) // 2
        node.left = SegmentTreeNode.build_segment_tree(
            arr, SegmentTreeNode(start, mid, node.dataoperation), start, mid
        )
        node.right = SegmentTreeNode.build_segment_tree(
            arr, SegmentTreeNode(mid + 1, end, node.dataoperation), mid + 1, end
        )
        node.value = node.dataoperation(
            node.left.value, node.right.value
        )  # 示例：求和，根据需要调整合并逻辑
        return node

    def update_segment_tree(node: "SegmentTreeNode", index, value):
        if node.start == node.end:  # 叶节点
            node.value = value
            return node.value
        mid = (node.start + node.end) // 2
        if index <= mid:
            SegmentTreeNode.update_segment_tree(node.left, index, value)
        else:
            SegmentTreeNode.update_segment_tree(node.right, index, value)
        node.value = node.dataoperation(
            node.left.value, node.right.value
        )  # 示例：求和，根据需要调整合并逻辑
        return node.value

    def query_segment_tree(node, start, end) -> int:
        if node.start >= start and node.end <= end:  # 当前节点完全在查询区间内
            return node.value
        if node.end < start or node.start > end:  # 当前节点完全在查询区间外
            return inf  # 或其他默认值，根据需要调整
        left_sum = (
            SegmentTreeNode.query_segment_tree(node.left, start, end)
            if node.left
            else inf
        )
        right_sum = (
            SegmentTreeNode.query_segment_tree(node.right, start, end)
            if node.right
            else inf
        )
        return node.dataoperation(
            left_sum, right_sum
        )  # 示例：求和，根据需要调整合并逻辑


class Pos:
    def __init__(self, l: int, t: int, r: int, b: int):
        self.l = l
        self.t = t
        self.r = r
        self.b = b


class Bank:

    def __init__(self, balance: List[int]):
        self.balance = balance
        self.size = len(balance)

    def transfer(self, account1: int, account2: int, money: int) -> bool:
        if account1 < 1 or account1 > self.size:
            return False
        if account2 < 1 or account2 > self.size:
            return False
        if money > self.balance[account1 - 1]:
            return False
        self.balance[account1 - 1] -= money
        self.balance[account2 - 1] += money
        return True

    def deposit(self, account: int, money: int) -> bool:
        if account < 1 or account > self.size:
            return False
        self.balance[account - 1] += money
        return True

    def withdraw(self, account: int, money: int) -> bool:
        if account < 1 or account > self.size:
            return False
        if self.balance[account - 1] < money:
            return False
        self.balance[account - 1] -= money
        return True
