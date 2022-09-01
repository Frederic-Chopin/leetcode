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