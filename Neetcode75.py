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