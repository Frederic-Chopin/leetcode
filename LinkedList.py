#Linked List
#143. Reorder List
#Time: O(N), Space: O(1)
def reorderList(self, head: Optional[ListNode]) -> None:
        #get midpoint pointer
        slow, fast = head, head.next
        while fast:
            slow = slow.next
            fast = fast.next
            if fast:
                fast = fast.next
        
        #reverse second half
        prev, curr = None, slow
        while curr:
            temp = curr.next
            curr.next = prev
            prev = curr
            curr = temp
        
        #construct new LL
        ptr = head
        while prev.next:
            p1, p2 = ptr.next, prev.next
            ptr.next = prev
            prev.next = p1
            ptr, prev = p1, p2

#143. Remove Nth Node From End of List
#Time: O(N), Space: O(1)
#dummy node, 2 pointers
def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        #1 pass
        dummy = ListNode(0,head)
        left, right = dummy, head
        while n > 0:
            right = right.next
            n -= 1
        while right:
            right = right.next
            left = left.next
        left.next = left.next.next
        return dummy.next