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