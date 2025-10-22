from Common import ListNode
from typing import Optional

class Solution1:

    def hello():
        print("wwww")
        return

    def removeElements_203(self, head: Optional[ListNode], val: int) -> Optional[ListNode]:
        if head is None:
            return head
        
        first = head
        pre: Optional[ListNode] = None
        while head != None:
            if head.val == val:
                if head.next != None:
                    head.val = head.next.val
                    head.next = head.next.next
                else:
                    if pre == None:
                        first = None
                    else:
                        pre.next = None
                    head = None
            else:
                pre = head 
                head = head.next
        
        return first
