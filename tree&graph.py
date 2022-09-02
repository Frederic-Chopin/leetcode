#Tree
#226. Invert Binary Tree
#Recursion
#Time: O(N), N:#of nodes; Space: O(h), h:height
def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return root
        root.left, root.right = self.invertTree(root.right), self.invertTree(root.left)
        return root


#104. Maximum Depth of Binary Tree
#Recursion
#Time: O(N); Space: O(N), worst case unbalanced
def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))
#Iteration, BFS
#deque: append, appendleft, pop, popleft, conut(x), insert(i,x), remove(x), reverse, extend('abc'), extendleft(iterable)
def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        
        fringe = collections.deque([root])
        level = 0
        while fringe:
            for i in range(len(fringe)):
                node = fringe.popleft()
                if node.left:
                    fringe.append(node.left)
                if node.right:
                    fringe.append(node.right)
            level += 1
        return level


#100. Same Tree
#Recursion
#Time: O(N); Space: O(N)
def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        #Recursion
        if not p and not q:
            return True
        if not p or not q:
            return False
        if p.val != q.val:
            return False
        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
#Iteration, DFS
def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        stack = [(p, q)]
        while stack:
            node1, node2 = stack.pop()      #pop(0), BFS; use deque popleft() is better. 
            if not node1 and not node2:
                continue
            elif None in [node1, node2]:
                return False
            else:
                if node1.val != node2.val:
                    return False
                stack.append((node1.right, node2.right))
                stack.append((node1.left, node2.left))
        return True


#572. Subtree of Another Tree
#Recursion
#Time: O(S*T); Space: O(S+T)
def isSubtree(self, s: Optional[TreeNode], t: Optional[TreeNode]) -> bool:
    def dfs(s, t):
        if not s and not t: return True
        if not s or not t: return False
        return s.val == t.val and dfs(s.left, t.left) and dfs(s.right, t.right)

    if not s: return
    if s.val == t.val and dfs(s, t): return True
    return self.isSubtree(s.left, t) or self.isSubtree(s.right, t)
#(Merkle hashing)
#O(|s| + |t|) 
def isSubtree(self, s, t):
    from hashlib import sha256
    def hash_(x):
        S = sha256()
        S.update(x)
        return S.hexdigest()
        
    def merkle(node):
        if not node:
            return '#'
        m_left = merkle(node.left)
        m_right = merkle(node.right)
        node.merkle = hash_(m_left + str(node.val) + m_right)
        return node.merkle
        
    merkle(s)
    merkle(t)
    def dfs(node):
        if not node:
            return False
        return (node.merkle == t.merkle or 
                dfs(node.left) or dfs(node.right))
                    
    return dfs(s)


#235. Lowest Common Ancestor of a Binary Search Tree
#Recursion
#Time: O(N); Space: O(N), worset case skewed
def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        parent_val = root.val
        p_val = p.val
        q_val = q.val

        if p_val > parent_val and q_val > parent_val:    
            return self.lowestCommonAncestor(root.right, p, q)
        elif p_val < parent_val and q_val < parent_val:    
            return self.lowestCommonAncestor(root.left, p, q)
        else:
            return root


#102. Binary Tree Level Order Traversal
#Iteration
#Time: O(N); Space: O(N), max # of leaf node is (n+1)/2 on last level if full
def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        fringe = collections.deque([root])
        res = []
        while fringe:
            count = len(fringe)
            level = []
            out = False
            for i in range(len(fringe)):
                node = fringe.popleft()
                level.append(node.val)
                if node.left:
                    fringe.append(node.left)
                if node.right:
                    fringe.append(node.right)
            res.append(level)
        return res
#Recursive
def levelOrder(self, root):
        levels = []
        if not root:
            return levels
        
        def helper(node, level):
            if len(levels) == level:
                levels.append([])

            levels[level].append(node.val)

            if node.left:
                helper(node.left, level + 1)
            if node.right:
                helper(node.right, level + 1)
            
        helper(root, 0)
        return levels


#98. Validate Binary Search Tree
#Iterative Inorder Traversal
#Time: O(N); Space: O(N)
def isValidBST(self, root: Optional[TreeNode]) -> bool:
    if not root:
        return True
    stack = []
    ptr = root
    prev = -float('inf')
    while stack or ptr:
        while ptr:
            stack.append(ptr)
            ptr = ptr.left
        ptr = stack.pop()
        if ptr.val <= prev:
            return False
        prev = ptr.val
        ptr = ptr.right
    return True
#Recursive inorder traversal
def isValidBST(self, root: TreeNode) -> bool:
    def inorder(root):
        if not root:
            return True
        if not inorder(root.left):
            return False
        if root.val <= self.prev:
            return False
        self.prev = root.val
        return inorder(root.right)
    self.prev = -math.inf
    return inorder(root)
#Recursion with valid range
def isValidBST(self, root: TreeNode) -> bool:
    def validate(node, low=-math.inf, high=math.inf):
        if not node:
            return True
        if node.val <= low or node.val >= high:
            return False
        return (validate(node.right, node.val, high) and
                validate(node.left, low, node.val))
    return validate(root)


#230. Kth Smallest Element in a BST
#Iterative DFS Inorder Traversal
#Time: O(N), Space: O(N)
def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
    count = 0
    ptr = root
    fringe = collections.deque()
    while ptr or fringe:
        while ptr:
            fringe.append(ptr)
            ptr = ptr.left
            
        ptr = fringe.pop() 
        count += 1
        if count == k:
            break
        ptr = ptr.right
    return ptr.val
#Recursive Inorder
def kthSmallest(self, root, k):
    def inorder(r):
        return inorder(r.left) + [r.val] + inorder(r.right) if r else []

    return inorder(root)[k - 1]
 
