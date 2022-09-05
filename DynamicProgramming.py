#Dynamic Programming
#213. House Robber II
#DP using iteration, 2 pointers
#Time: O(N); Space: O(1)
def rob(self, nums: List[int]) -> int:
        if len(nums) == 0 or nums is None:
            return 0

        if len(nums) == 1:
            return nums[0]

        return max(self.rob_simple(nums[:-1]), self.rob_simple(nums[1:]))

def rob_simple(self, nums: List[int]) -> int:
    t1 = 0
    t2 = 0
    for current in nums:
        temp = t1
        t1 = max(current + t2, t1)
        t2 = temp

    return t1


#5. Longest Palindromic Substring
#Expand Around Center
#Time: O(N^2); Space: O(1)
def longestPalindrome(self, s: str) -> str:
        if not s:
            return ""
        i, max_len = 0, 1
        res = s[0]
        while i < len(s) - 1:
            left, right = i, i
            curr_len = 1
            while right+1 < len(s) and s[right+1] == s[i]:
                right += 1
                curr_len += 1
            while left-1 >= 0 and right+1 < len(s) and s[left-1] == s[right+1]:
                left, right = left - 1, right + 1
                curr_len += 2
            if curr_len > max_len:
                res = s[left:right+1]
                max_len = curr_len
            i = i + 1
        return res
#Recursion                              
def longestPalindrome(self, s):
    res = ""
    for i in range(len(s)):
        tmp = self.helper(s, i, i)
        if len(tmp) > len(res):
            res = tmp
        tmp = self.helper(s, i, i+1)
        if len(tmp) > len(res):
            res = tmp
    return res
def helper(self, s, l, r):
    while l >= 0 and r < len(s) and s[l] == s[r]:
        l -= 1; r += 1
    return s[l+1:r]