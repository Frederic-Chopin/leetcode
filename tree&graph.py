#Tree
#Recursion
#Time: O(N), N:#of nodes; Space: O(h), h:height
def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return root
        
        root.left, root.right = self.invertTree(root.right), self.invertTree(root.left)

        return root