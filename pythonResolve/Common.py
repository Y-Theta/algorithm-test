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