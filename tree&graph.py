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