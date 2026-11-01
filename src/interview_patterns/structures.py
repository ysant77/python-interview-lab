"""Linked lists, trees, traversal, graph search and data structures.

All algorithms are handwritten Python reference implementations; no source repo
code was copied. Auxiliary-space costs omit answer storage unless noted.
"""
from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass
from heapq import heappush, heappop
from typing import Optional

@dataclass(eq=False)
class ListNode:
    val: int = 0
    next: Optional['ListNode'] = None

@dataclass(eq=False)
class TreeNode:
    val: int = 0
    left: Optional['TreeNode'] = None
    right: Optional['TreeNode'] = None


def has_cycle(head: Optional[ListNode]) -> bool:
    """LC 141. Floyd two-speed runners. O(n) time, O(1) space."""
    slow=fast=head
    while fast and fast.next:
        slow=slow.next;fast=fast.next.next
        if slow is fast:return True
    return False


def detect_cycle(head: Optional[ListNode]) -> Optional[ListNode]:
    """LC 142. After meeting, advance finder and slow together. O(n), O(1)."""
    slow=fast=head
    while fast and fast.next:
        slow=slow.next;fast=fast.next.next
        if slow is fast:
            finder=head
            while finder is not slow:finder=finder.next;slow=slow.next
            return finder
    return None


def middle_node(head: Optional[ListNode]) -> Optional[ListNode]:
    """LC 876. Second middle for even length. O(n), O(1)."""
    slow=fast=head
    while fast and fast.next:slow=slow.next;fast=fast.next.next
    return slow


def reverse_list(head: Optional[ListNode]) -> Optional[ListNode]:
    """LC 206. Rewire one pointer per node. O(n), O(1)."""
    prev=None
    while head:
        nxt=head.next;head.next=prev;prev=head;head=nxt
    return prev


def reverse_between(head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
    """LC 92. Reverse one-indexed inclusive sublist. O(n), O(1)."""
    dummy=ListNode(next=head);before=dummy
    for _ in range(left-1):before=before.next
    curr=before.next
    for _ in range(right-left):
        moving=curr.next;curr.next=moving.next;moving.next=before.next;before.next=moving
    return dummy.next


def reverse_k_group(head: Optional[ListNode], k: int) -> Optional[ListNode]:
    """LC 25. Only reverse complete groups. O(n), O(1)."""
    if k<=0:raise ValueError('k must be positive')
    dummy=ListNode(next=head);before=dummy
    while True:
        kth=before
        for _ in range(k):
            kth=kth.next
            if kth is None:return dummy.next
        after=kth.next;prev=after;cur=before.next
        while cur is not after:
            nxt=cur.next;cur.next=prev;prev=cur;cur=nxt
        old_first=before.next;before.next=kth;before=old_first


def merge_two_lists(a: Optional[ListNode], b: Optional[ListNode]) -> Optional[ListNode]:
    """LC 21. Reuse nodes; sorted lists. O(n+m), O(1)."""
    dummy=tail=ListNode()
    while a and b:
        if a.val<=b.val:tail.next=a;a=a.next
        else:tail.next=b;b=b.next
        tail=tail.next
    tail.next=a or b
    return dummy.next


def swap_pairs(head: Optional[ListNode]) -> Optional[ListNode]:
    """LC 24. Pointer rewiring in pairs. O(n), O(1)."""
    dummy=ListNode(next=head);prev=dummy
    while prev.next and prev.next.next:
        a=prev.next;b=a.next;a.next=b.next;b.next=a;prev.next=b;prev=a
    return dummy.next


def remove_nth_from_end(head: Optional[ListNode], n: int) -> Optional[ListNode]:
    """LC 19. Maintain n-node fast/slow separation. O(n), O(1)."""
    dummy=ListNode(next=head);fast=slow=dummy
    for _ in range(n):
        fast=fast.next
        if fast is None:raise ValueError('n exceeds list length')
    while fast.next:fast=fast.next;slow=slow.next
    slow.next=slow.next.next
    return dummy.next


def is_palindrome_list(head: Optional[ListNode]) -> bool:
    """LC 234. Materialize values for clarity. O(n) time/space."""
    values=[]
    while head:values.append(head.val);head=head.next
    return values==values[::-1]


def get_intersection_node(a: Optional[ListNode], b: Optional[ListNode]) -> Optional[ListNode]:
    """LC 160. Two runners switch lists. O(n+m), O(1)."""
    p,q=a,b
    while p is not q:
        p=p.next if p else b
        q=q.next if q else a
    return p


def max_depth(root: Optional[TreeNode]) -> int:
    """LC 104. DFS depth. O(n) time, O(h) stack."""
    return 0 if root is None else 1+max(max_depth(root.left),max_depth(root.right))


def diameter_of_binary_tree(root: Optional[TreeNode]) -> int:
    """LC 543. Longest edge-count path through any node. O(n), O(h)."""
    best=0
    def dfs(node):
        nonlocal best
        if node is None:return 0
        l=dfs(node.left);r=dfs(node.right)
        best=max(best,l+r);return max(l,r)+1
    dfs(root);return best


def invert_tree(root: Optional[TreeNode]) -> Optional[TreeNode]:
    """LC 226. Swap children recursively. O(n) time, O(h) space."""
    if root:
        root.left,root.right=invert_tree(root.right),invert_tree(root.left)
    return root


def has_path_sum(root: Optional[TreeNode], target: int) -> bool:
    """LC 112. Root-to-leaf target path. O(n), O(h)."""
    if not root:return False
    if not root.left and not root.right:return root.val==target
    return has_path_sum(root.left,target-root.val) or has_path_sum(root.right,target-root.val)


def path_sum_ii(root: Optional[TreeNode], target: int) -> list[list[int]]:
    """LC 113. Backtracking root-to-leaf paths. O(n*h + output), O(h) stack."""
    out=[];path=[]
    def visit(node,rest):
        if node is None:return
        path.append(node.val);rest-=node.val
        if not node.left and not node.right and rest==0:out.append(path[:])
        else:visit(node.left,rest);visit(node.right,rest)
        path.pop()
    visit(root,target);return out


def max_path_sum(root: Optional[TreeNode]) -> int:
    """LC 124. Drop negative branch gain. O(n), O(h)."""
    if not root:raise ValueError('root required')
    best=float('-inf')
    def gain(node):
        nonlocal best
        if not node:return 0
        left=max(0,gain(node.left));right=max(0,gain(node.right))
        best=max(best,node.val+left+right)
        return node.val+max(left,right)
    gain(root);return int(best)


def level_order(root: Optional[TreeNode]) -> list[list[int]]:
    """LC 102. Breadth first queue. O(n) time, O(w) working space."""
    if not root:return []
    q=deque([root]);out=[]
    while q:
        level=[]
        for _ in range(len(q)):
            node=q.popleft();level.append(node.val)
            if node.left:q.append(node.left)
            if node.right:q.append(node.right)
        out.append(level)
    return out


def zigzag_level_order(root: Optional[TreeNode]) -> list[list[int]]:
    """LC 103. BFS with alternate levels. O(n), O(w)."""
    result=level_order(root)
    for i in range(1,len(result),2):result[i].reverse()
    return result


def right_side_view(root: Optional[TreeNode]) -> list[int]:
    """LC 199. Last element per BFS level. O(n), O(w)."""
    return [row[-1] for row in level_order(root)]


def is_valid_bst(root: Optional[TreeNode]) -> bool:
    """LC 98. Carry strict bounds down the tree. O(n), O(h)."""
    def dfs(node,lo,hi):
        if node is None:return True
        if not lo<node.val<hi:return False
        return dfs(node.left,lo,node.val) and dfs(node.right,node.val,hi)
    return dfs(root,float('-inf'),float('inf'))


def search_bst(root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
    """LC 700. Sorted BST property. O(h), O(1)."""
    while root and root.val!=val:
        root=root.left if val<root.val else root.right
    return root


def kth_smallest_bst(root: Optional[TreeNode], k: int) -> int:
    """LC 230. Iterative in-order. O(h+k) time, O(h) space."""
    st=[];node=root
    while st or node:
        while node:st.append(node);node=node.left
        node=st.pop();k-=1
        if k==0:return node.val
        node=node.right
    raise ValueError('k exceeds node count')


def num_islands(grid: list[list[str]]) -> int:
    """LC 200. Flood fill. O(rows*cols) time/space; grid not mutated."""
    if not grid:return 0
    rows,cols=len(grid),len(grid[0]);visited=set();count=0
    for i in range(rows):
        for j in range(cols):
            if grid[i][j]!='1' or (i,j) in visited:continue
            count+=1;stack=[(i,j)];visited.add((i,j))
            while stack:
                r,c=stack.pop()
                for nr,nc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                    if 0<=nr<rows and 0<=nc<cols and grid[nr][nc]=='1' and (nr,nc) not in visited:
                        visited.add((nr,nc));stack.append((nr,nc))
    return count


def max_area_of_island(grid: list[list[int]]) -> int:
    """LC 695. Iterative DFS; no mutation. O(RC) time/space."""
    if not grid:return 0
    rows,cols=len(grid),len(grid[0]);seen=set();best=0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c]!=1 or (r,c) in seen:continue
            stack=[(r,c)];seen.add((r,c));area=0
            while stack:
                x,y=stack.pop();area+=1
                for a,b in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
                    if 0<=a<rows and 0<=b<cols and grid[a][b]==1 and (a,b) not in seen:
                        seen.add((a,b));stack.append((a,b))
            best=max(best,area)
    return best


def flood_fill(image: list[list[int]], sr: int, sc: int, color: int) -> list[list[int]]:
    """LC 733. BFS; mutates image. O(RC) time, O(RC) queue worst case."""
    original=image[sr][sc]
    if original==color:return image
    q=deque([(sr,sc)]);image[sr][sc]=color
    while q:
        r,c=q.popleft()
        for x,y in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
            if 0<=x<len(image) and 0<=y<len(image[0]) and image[x][y]==original:
                image[x][y]=color;q.append((x,y))
    return image


def oranges_rotting(grid: list[list[int]]) -> int:
    """LC 994. Multi-source BFS; mutates input. O(RC), O(RC) space."""
    if not grid:return 0
    rows,cols=len(grid),len(grid[0]);q=deque();fresh=0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c]==2:q.append((r,c,0))
            elif grid[r][c]==1:fresh+=1
    elapsed=0
    while q:
        r,c,minute=q.popleft();elapsed=max(elapsed,minute)
        for x,y in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
            if 0<=x<rows and 0<=y<cols and grid[x][y]==1:
                grid[x][y]=2;fresh-=1;q.append((x,y,minute+1))
    return elapsed if fresh==0 else -1


def can_finish(num_courses: int, prerequisites: list[list[int]]) -> bool:
    """LC 207. Kahn's DAG test. O(V+E), O(V+E)."""
    return len(find_order(num_courses,prerequisites))==num_courses


def find_order(num_courses: int, prerequisites: list[list[int]]) -> list[int]:
    """LC 210. Topological order or [] on cycle. O(V+E), O(V+E)."""
    adj=[[] for _ in range(num_courses)];degree=[0]*num_courses
    for course,pre in prerequisites:adj[pre].append(course);degree[course]+=1
    q=deque(i for i,x in enumerate(degree) if x==0);out=[]
    while q:
        node=q.popleft();out.append(node)
        for nxt in adj[node]:
            degree[nxt]-=1
            if degree[nxt]==0:q.append(nxt)
    return out if len(out)==num_courses else []


class DSU:
    """Path compression + union by size. Amortized O(alpha(n)) per operation."""
    def __init__(self,n:int):self.parent=list(range(n));self.size=[1]*n;self.components=n
    def find(self,x:int)->int:
        while self.parent[x]!=x:
            self.parent[x]=self.parent[self.parent[x]];x=self.parent[x]
        return x
    def union(self,a:int,b:int)->bool:
        a=self.find(a);b=self.find(b)
        if a==b:return False
        if self.size[a]<self.size[b]:a,b=b,a
        self.parent[b]=a;self.size[a]+=self.size[b];self.components-=1;return True


def find_circle_num(is_connected: list[list[int]]) -> int:
    """LC 547. Union connected cities. O(n^2 alpha(n)), O(n)."""
    n=len(is_connected);dsu=DSU(n)
    for i in range(n):
        for j in range(i+1,n):
            if is_connected[i][j]:dsu.union(i,j)
    return dsu.components


def redundant_connection(edges: list[list[int]]) -> list[int]:
    """LC 684. First edge forming a cycle. O(E alpha(V)), O(V)."""
    dsu=DSU(max(max(e) for e in edges)+1)
    for u,v in edges:
        if not dsu.union(u,v):return [u,v]
    return []


def network_delay_time(times: list[list[int]], n: int, k: int) -> int:
    """LC 743. Dijkstra; positive edges. O((V+E) log V), O(V+E)."""
    adj=defaultdict(list)
    for u,v,w in times:adj[u].append((v,w))
    best={k:0};heap=[(0,k)]
    while heap:
        d,u=heappop(heap)
        if d!=best[u]:continue
        for v,w in adj[u]:
            alt=d+w
            if alt<best.get(v,float('inf')):
                best[v]=alt;heappush(heap,(alt,v))
    return max(best.values()) if len(best)==n else -1


def min_cost_connect_points(points: list[list[int]]) -> int:
    """LC 1584. Prim on implicit complete Manhattan graph. O(n^2), O(n)."""
    n=len(points)
    if not n:return 0
    best=[float('inf')]*n;best[0]=0;used=[False]*n;cost=0
    for _ in range(n):
        u=min((i for i in range(n) if not used[i]),key=lambda i:best[i])
        used[u]=True;cost+=best[u]
        for v in range(n):
            if not used[v]:
                d=abs(points[u][0]-points[v][0])+abs(points[u][1]-points[v][1]);best[v]=min(best[v],d)
    return int(cost)


class Trie:
    """LC 208. Prefix tree. Insert/search O(L) time, O(total chars) storage."""
    def __init__(self):self.root={}
    def insert(self,word:str)->None:
        cur=self.root
        for ch in word:cur=cur.setdefault(ch,{})
        cur['#']=True
    def search(self,word:str)->bool:
        cur=self._find(word)
        return bool(cur is not None and '#' in cur)
    def starts_with(self,prefix:str)->bool:
        return self._find(prefix) is not None
    def _find(self,s):
        cur=self.root
        for ch in s:
            if ch not in cur:return None
            cur=cur[ch]
        return cur


def subsets(nums: list[int]) -> list[list[int]]:
    """LC 78. Double results on each item. O(n*2^n) time/output."""
    out=[[]]
    for x in nums:out += [row+[x] for row in out]
    return out


def subsets_with_dup(nums: list[int]) -> list[list[int]]:
    """LC 90. Sort and extend only new subsets for duplicated value. O(n 2^n)."""
    nums=sorted(nums);out=[[]];start=0
    for i,x in enumerate(nums):
        old_len=len(out);begin=start if i>0 and nums[i]==nums[i-1] else 0
        out.extend(out[j]+[x] for j in range(begin,old_len));start=old_len
    return out


def permute(nums: list[int]) -> list[list[int]]:
    """LC 46. Backtrack with used mask. O(n*n!), O(n) work plus output."""
    out=[];path=[];used=[False]*len(nums)
    def dfs():
        if len(path)==len(nums):out.append(path[:]);return
        for i,x in enumerate(nums):
            if used[i]:continue
            used[i]=True;path.append(x);dfs();path.pop();used[i]=False
    dfs();return out


def combine(n: int,k: int) -> list[list[int]]:
    """LC 77. Generate increasing k-combinations. O(k*C(n,k)) output."""
    out=[];path=[]
    def dfs(start):
        if len(path)==k:out.append(path[:]);return
        for x in range(start,n+1):
            path.append(x);dfs(x+1);path.pop()
    dfs(1);return out


def combination_sum(candidates: list[int],target: int) -> list[list[int]]:
    """LC 39. Unlimited reuse, positive distinct candidates. Exponential, O(target/min) stack."""
    candidates=sorted(candidates);out=[];path=[]
    def dfs(start,remaining):
        if remaining==0:out.append(path[:]);return
        for i in range(start,len(candidates)):
            x=candidates[i]
            if x>remaining:break
            path.append(x);dfs(i,remaining-x);path.pop()
    dfs(0,target);return out


def word_search(board: list[list[str]],word: str) -> bool:
    """LC 79. No cell reused. Worst O(RC*3^L), O(L) recursion/visited."""
    if not word:return True
    if not board:return False
    rows,cols=len(board),len(board[0]);used=set()
    def dfs(r,c,i):
        if i==len(word):return True
        if not(0<=r<rows and 0<=c<cols) or (r,c) in used or board[r][c]!=word[i]:return False
        used.add((r,c))
        found=any(dfs(x,y,i+1) for x,y in ((r+1,c),(r-1,c),(r,c+1),(r,c-1)))
        used.remove((r,c));return found
    return any(dfs(r,c,0) for r in range(rows) for c in range(cols))


def shortest_path_binary_matrix(grid:list[list[int]])->int:
    """LC 1091. Eight-direction BFS. O(RC) time/space, no input mutation."""
    if not grid or not grid[0] or grid[0][0] or grid[-1][-1]:return -1
    rows=len(grid);cols=len(grid[0]);q=deque([(0,0,1)]);seen={(0,0)}
    while q:
        r,c,d=q.popleft()
        if (r,c)==(rows-1,cols-1):return d
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                if dr==dc==0:continue
                x,y=r+dr,c+dc
                if 0<=x<rows and 0<=y<cols and grid[x][y]==0 and (x,y) not in seen:
                    seen.add((x,y));q.append((x,y,d+1))
    return -1


def closed_island(grid:list[list[int]])->int:
    """LC 1254. Count zero-connected components not touching boundary. O(RC), O(RC)."""
    rows=len(grid);cols=len(grid[0]) if rows else 0;seen=set();count=0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c]!=0 or (r,c) in seen:continue
            stack=[(r,c)];seen.add((r,c));closed=True
            while stack:
                x,y=stack.pop()
                if x in (0,rows-1) or y in (0,cols-1):closed=False
                for a,b in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
                    if 0<=a<rows and 0<=b<cols and grid[a][b]==0 and (a,b) not in seen:
                        seen.add((a,b));stack.append((a,b))
            count+=closed
    return count
