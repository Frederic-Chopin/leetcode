#Binary Search
#153. (Medium) Find Minimum in Rotated Sorted Array
#Time: O(logN), Space: O(1)
def findMin(self, nums: List[int]) -> int:
        left, right = 0, len(nums)-1
        while left <= right:
            mid = (left+right) // 2
            if nums[mid-1] > nums[mid]:     #found the pivoting point
                return nums[mid]
            if nums[mid] > nums[right]:   #in the left half
                left = mid + 1
            else:
                right = mid - 1         #in the right half
        return nums[0]
def findMin(self, nums: List[int]) -> int:
        left, right = 0, len(nums)-1
        while left < right:
            mid = (left + right) // 2
            if nums[mid] > nums[right]:
                left = mid + 1
            else:
                right = mid
        return nums[left]

#Linked List
#143. Reorder List
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

        