from typing import Optional,List


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
    def dotProduct(self, vec: 'SparseVector') -> int:
        interct = self._keyset.intersection(vec._keyset)
        sum = 0
        for i in interct:
            sum += self._dict[i] * vec._dict[i]
        return sum