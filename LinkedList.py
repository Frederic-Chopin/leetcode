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

#19. Remove Nth Node From End of List
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

#141. Linked List Cycle
#Time: O(N), Space: O(1)
#Floyd's Tortoise & Hare, 2 pointers
def hasCycle(self, head: Optional[ListNode]) -> bool:
    slow, fast = head, head
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
        if slow == fast:
            return True
    return False

#23. Merge K Sorted Lists
#Time: O(N*logK), N nodes, K lists; Space: O(1)
#Merge Sort, also min heap (T: O(N*logK), S: O(N))
def mergeKLists(self, lists):
        amount = len(lists)
        interval = 1
        while interval < amount:
            for i in range(0, amount - interval, interval * 2):
                lists[i] = self.merge2Lists(lists[i], lists[i + interval])
            interval *= 2
        return lists[0] if amount > 0 else None

    def merge2Lists(self, l1, l2):
        head = point = ListNode(0)
        while l1 and l2:
            if l1.val <= l2.val:
                point.next = l1
                l1 = l1.next
            else:
                point.next = l2
                l2 = l1
                l1 = point.next.next
            point = point.next
        if not l1:
            point.next=l2
        else:
            point.next=l1
        return head.next

