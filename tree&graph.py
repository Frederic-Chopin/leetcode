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
 

 #450. Delete Node in a BST
 #Recursion: 1. node is leaf: null; 
 #           2. node not leaf, has right child, replaced by succesor; 
 #           3. node not leaf, has left child but no right child, replaced by predecessor.
 #Time: O(logN); Space: O(H), H = logN
 def deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:
        if not root: return None
        if root.val == key:
            if root.left:
                # Find the right most leaf of the left sub-tree
                left_right_most = root.left
                while left_right_most.right:
                    left_right_most = left_right_most.right
                # Attach right child to the right of that leaf
                left_right_most.right = root.right
                # Return left child instead of root, a.k.a delete root
                return root.left
            else:
                return root.right
        # If left or right child got deleted, the returned root is the child of the deleted node.
        elif root.val > key:
            root.left = self.deleteNode(root.left, key)
        else:
            root.right = self.deleteNode(root.right, key)
            
        return root


#105. Construct Binary Tree from Preorder and Inorder Traversal
#Recursion: 
# preorder[0] is root, 
# left of inorder is root.left, right of inorder is root.right
#Time: O(N); Space: O(N)
def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
    if not preorder or not inorder:
        return None
    root = TreeNode(preorder[0])
    mid = inorder.index(preorder[0])
    root.left = self.buildTree(preorder[1:mid+1], inorder[:mid])
    root.right = self.buildTree(preorder[mid+1:], inorder[mid+1:])
    return root



#Graphs
#133. Clone Graph
#DFS Iteration
#Time: O(N+M); Space: O(N)
def cloneGraph(self, node: 'Node') -> 'Node':
        #null case
        if not node:
            return None
        
        #DFS
        ptr = node
        fringe = collections.deque([ptr])
        mp = {}
        mp[ptr.val] = Node(ptr.val)
        
        while fringe:
            ptr = fringe.pop()
            for nb in ptr.neighbors:
                if nb.val not in mp:
                    mp[nb.val] = Node(nb.val)
                    fringe.append(nb)
                mp[ptr.val].neighbors.append(mp[nb.val])
                    
        return mp[node.val]

#Recursion
def __init__(self):
        self.visited = {}

def cloneGraph(self, node):
    if not node:
        return node

    if node in self.visited:
        return self.visited[node]
    
    clone_node = Node(node.val, [])
    self.visited[node] = clone_node
    
    if node.neighbors:
        clone_node.neighbors = [self.cloneGraph(n) for n in node.neighbors]
    return clone_node
                    


#417. Pacific Atlantic Water Flow
#Run BFS from the ocean (boarders), Iteration
#Time: O(MN); Space: (MN)
def pacificAtlantic(self, matrix: List[List[int]]) -> List[List[int]]:
    if not matrix or not matrix[0]: 
        return []
        
    num_rows, num_cols = len(matrix), len(matrix[0])

    pacific_queue = deque()
    atlantic_queue = deque()
    for i in range(num_rows):
        pacific_queue.append((i, 0))
        atlantic_queue.append((i, num_cols - 1))
    for i in range(num_cols):
        pacific_queue.append((0, i))
        atlantic_queue.append((num_rows - 1, i))
    
    def bfs(queue):
        reachable = set()
        while queue:
            (row, col) = queue.popleft()
            reachable.add((row, col))
            for (x, y) in [(1, 0), (0, 1), (-1, 0), (0, -1)]: 
                new_row, new_col = row + x, col + y
                if new_row < 0 or new_row >= num_rows or new_col < 0 or new_col >= num_cols:
                    continue
                if (new_row, new_col) in reachable:
                    continue
                if matrix[new_row][new_col] < matrix[row][col]:
                    continue
                queue.append((new_row, new_col))
        return reachable

    pacific_reachable = bfs(pacific_queue)
    atlantic_reachable = bfs(atlantic_queue)

    return list(pacific_reachable.intersection(atlantic_reachable))

#DFS, Recursion
def pacificAtlantic(self, matrix: List[List[int]]) -> List[List[int]]:
    if not matrix or not matrix[0]: 
        return []

    num_rows, num_cols = len(matrix), len(matrix[0])
    pacific_reachable = set()
    atlantic_reachable = set()
    
    def dfs(row, col, reachable):
        reachable.add((row, col))
        for (x, y) in [(1, 0), (0, 1), (-1, 0), (0, -1)]: 
            new_row, new_col = row + x, col + y
            if new_row < 0 or new_row >= num_rows or new_col < 0 or new_col >= num_cols:
                continue
            if (new_row, new_col) in reachable:
                continue
            if matrix[new_row][new_col] < matrix[row][col]:
                continue
            dfs(new_row, new_col, reachable)
    
    for i in range(num_rows):
        dfs(i, 0, pacific_reachable)
        dfs(i, num_cols - 1, atlantic_reachable)
    for i in range(num_cols):
        dfs(0, i, pacific_reachable)
        dfs(num_rows - 1, i, atlantic_reachable)

    return list(pacific_reachable.intersection(atlantic_reachable))



#323. Number of Connected Components in an Undirected Graph
#Union Find using disjoint set
#Time: O(E*a(n)) < O(n*logn), a(n): inverse Ackermann function; Space: O(V)
def countComponents(self, n: int, edges: List[List[int]]) -> int:
    parents = [i for i in range(n)]
    rank = [1] * n

    def find(n1):
        res = n1
        while res != parents[res]:
            parents[res] = parents[parents[res]]
            res = parents[res]
        return res

    def union(n1, n2):
        p1, p2 = find(n1), find(n2)
        if p1 == p2: return 0
        if rank[p2] > rank[p1]:
            parents[p1] = p2
            rank[p2] += rank[p1]
        else:
            parents[p2] = p1
            rank[p1] += rank[p2]
        return 1

    res = n
    for n1, n2 in edges:
        res -= union(n1, n2)
    return res

#DFS
#Time: O(E+V); Space: O(E+V)
def countComponents(self, n: int, edges: List[List[int]]) -> int:
    graph = collections.defaultdict(list)
    for x, y in edges:
        graph[x].append(y)
        graph[y].append(x)
    
    def dfs(node, seen):
        seen.add(node)
        for neighbor in graph[node]:
            if neighbor not in seen:
                dfs(neighbor, seen)
    count = 0
    seen = set()
    for node in range(n):
        if node not in seen:
            dfs(node, seen)
            count += 1
    return count
