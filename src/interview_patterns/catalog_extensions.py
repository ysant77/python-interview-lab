"""Solutions for the remaining workbook-indexed advanced DSA exercises.

All functions are independently written for this practice repository. They
prioritize transparent algorithms over micro-optimization; see LC IDs below.
"""
from __future__ import annotations
from collections import Counter, defaultdict, deque
from heapq import heappop, heappush
from math import inf
from typing import Optional
from .structures import TreeNode, DSU


def letter_combinations(digits:str)->list[str]:
    """LC 17 | Cartesian-product backtracking. T O(4^n*n), S O(n) aux."""
    keys={'2':'abc','3':'def','4':'ghi','5':'jkl','6':'mno','7':'pqrs','8':'tuv','9':'wxyz'}
    if not digits:return []
    if any(d not in keys for d in digits):raise ValueError('digits must be 2..9')
    result=[]
    def dfs(i,path):
        if i==len(digits):result.append(''.join(path));return
        for c in keys[digits[i]]:path.append(c);dfs(i+1,path);path.pop()
    dfs(0,[]);return result


def total_n_queens(n:int)->int:
    """LC 52 | Search rows with column/diagonal sets. Worst T O(n!), S O(n)."""
    if n<0:raise ValueError('n must be nonnegative')
    columns=set();diag1=set();diag2=set()
    def dfs(row):
        if row==n:return 1
        count=0
        for col in range(n):
            if col in columns or row-col in diag1 or row+col in diag2:continue
            columns.add(col);diag1.add(row-col);diag2.add(row+col)
            count+=dfs(row+1)
            columns.remove(col);diag1.remove(row-col);diag2.remove(row+col)
        return count
    return dfs(0)


def recover_tree(root:Optional[TreeNode])->None:
    """LC 99 | Identify two swapped inorder values; T O(n), S O(height)."""
    first=second=prev=None
    def walk(node):
        nonlocal first,second,prev
        if node is None:return
        walk(node.left)
        if prev is not None and prev.val>node.val:
            if first is None:first=prev
            second=node
        prev=node
        walk(node.right)
    walk(root)
    if first is not None and second is not None:first.val,second.val=second.val,first.val


def single_number_ii(nums:list[int])->int:
    """LC 137 | 32-bit bit-count residues; T O(32*n), S O(1)."""
    value=0
    for bit in range(32):
        if sum((number>>bit)&1 for number in nums)%3:value|=1<<bit
    return value-(1<<32) if value>=1<<31 else value


class WordDictionary:
    """LC 211 | Trie + wildcard '.' DFS. Add O(L), search worst O(26^L)."""
    def __init__(self):self.root={}
    def addWord(self,word:str)->None:
        node=self.root
        for ch in word:node=node.setdefault(ch,{})
        node['#']=True
    def search(self,word:str)->bool:
        def dfs(node,i):
            if i==len(word):return '#' in node
            char=word[i]
            return any(dfs(child,i+1) for ch,child in node.items() if ch!='#') if char=='.' else char in node and dfs(node[char],i+1)
        return dfs(self.root,0)


def find_words(board:list[list[str]],words:list[str])->list[str]:
    """LC 212 | Trie-guided DFS with pruning; T O(R*C*4^L + sum word lengths) worst."""
    if not board or not board[0] or not words:return []
    root={}
    for word in words:
        node=root
        for char in word:node=node.setdefault(char,{})
        node['$']=word
    rlen,clen=len(board),len(board[0]);found=[]
    def dfs(r,c,node):
        ch=board[r][c]
        if ch not in node:return
        child=node[ch]
        word=child.pop('$',None)
        if word is not None:found.append(word)
        board[r][c]='\0'
        for nr,nc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
            if 0<=nr<rlen and 0<=nc<clen and board[nr][nc]!='\0':dfs(nr,nc,child)
        board[r][c]=ch
        if not child:node.pop(ch)
    for r in range(rlen):
        for c in range(clen):dfs(r,c,root)
    return found


def lowest_common_ancestor_bst(root:Optional[TreeNode],p:TreeNode,q:TreeNode)->Optional[TreeNode]:
    """LC 235 | BST order determines LCA. T O(height), S O(1)."""
    while root:
        if p.val<root.val and q.val<root.val:root=root.left
        elif p.val>root.val and q.val>root.val:root=root.right
        else:return root
    return None


def count_bits(n:int)->list[int]:
    """LC 338 | DP: bits[i]=bits[i>>1]+(i&1). T/S O(n)."""
    if n<0:raise ValueError('n must be nonnegative')
    ans=[0]*(n+1)
    for i in range(1,n+1):ans[i]=ans[i>>1]+(i&1)
    return ans


def delete_node_bst(root:Optional[TreeNode],key:int)->Optional[TreeNode]:
    """LC 450 | Replace node with inorder successor. T O(height), S O(height)."""
    if root is None:return None
    if key<root.val:root.left=delete_node_bst(root.left,key)
    elif key>root.val:root.right=delete_node_bst(root.right,key)
    else:
        if root.left is None:return root.right
        if root.right is None:return root.left
        successor=root.right
        while successor.left:successor=successor.left
        root.val=successor.val
        root.right=delete_node_bst(root.right,successor.val)
    return root


def replace_words(dictionary:list[str],sentence:str)->str:
    """LC 648 | Trie prefix lookup per word. T O(sum roots + input chars)."""
    root={}
    for word in dictionary:
        node=root
        for ch in word:node=node.setdefault(ch,{})
        node['#']=True
    def replace(word):
        node=root;prefix=''
        for char in word:
            if '#' in node:return prefix
            if char not in node:return word
            node=node[char];prefix+=char
        return prefix if '#' in node else word
    return ' '.join(replace(w) for w in sentence.split())


class MapSum:
    """LC 677 | Prefix aggregate map with overwrite delta; insert O(L), sum O(L)."""
    def __init__(self):self.root={};self.values={}
    def insert(self,key:str,val:int)->None:
        delta=val-self.values.get(key,0);self.values[key]=val
        node=self.root
        node['total']=node.get('total',0)+delta
        for ch in key:
            node=node.setdefault(ch,{})
            node['total']=node.get('total',0)+delta
    def sum(self,prefix:str)->int:
        node=self.root
        for ch in prefix:
            if ch not in node:return 0
            node=node[ch]
        return node.get('total',0)


def swim_in_water(grid:list[list[int]])->int:
    """LC 778 | Minimax Dijkstra. T O(N² log N), S O(N²) for N×N grid."""
    if not grid or not grid[0]:raise ValueError('nonempty grid required')
    rows,cols=len(grid),len(grid[0]);pq=[(grid[0][0],0,0)];seen=set()
    while pq:
        level,r,c=heappop(pq)
        if (r,c) in seen:continue
        if (r,c)==(rows-1,cols-1):return level
        seen.add((r,c))
        for nr,nc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
            if 0<=nr<rows and 0<=nc<cols and (nr,nc) not in seen:heappush(pq,(max(level,grid[nr][nc]),nr,nc))
    raise ValueError('unreachable')


def cheapest_flights(n:int,flights:list[list[int]],src:int,dst:int,k:int)->int:
    """LC 787 | Bellman-Ford limited to k+1 edges. T O((k+1)*E), S O(n)."""
    dist=[inf]*n;dist[src]=0
    for _ in range(k+1):
        nxt=dist.copy()
        for a,b,c in flights:
            if dist[a]!=inf:nxt[b]=min(nxt[b],dist[a]+c)
        dist=nxt
    return -1 if dist[dst]==inf else int(dist[dst])


def shortest_subarray_at_least_k(nums:list[int],k:int)->int:
    """LC 862 | Monotonic prefix-sum deque with negatives. T/S O(n)."""
    prefix=0;dq=deque([(0,0)]);best=len(nums)+1
    for i,value in enumerate(nums,1):
        prefix+=value
        while dq and prefix-dq[0][1]>=k:
            best=min(best,i-dq.popleft()[0])
        while dq and dq[-1][1]>=prefix:dq.pop()
        dq.append((i,prefix))
    return best if best<=len(nums) else -1


def equations_possible(equations:list[str])->bool:
    """LC 990 | DSU equalities then inequalities. T O(n*alpha(26)), S O(1)."""
    uf=DSU(26)
    for expr in equations:
        if expr[1:3]=='==':uf.union(ord(expr[0])-97,ord(expr[3])-97)
    return all(uf.find(ord(e[0])-97)!=uf.find(ord(e[3])-97) for e in equations if e[1:3]=='!=')


def maximum_minimum_path(grid:list[list[int]])->int:
    """LC 1102 | Max-heap best bottleneck route. T O(RC log(RC)), S O(RC)."""
    rows,cols=len(grid),len(grid[0]);pq=[(-grid[0][0],0,0)];seen=set()
    while pq:
        neg,r,c=heappop(pq)
        if (r,c) in seen:continue
        score=-neg
        if (r,c)==(rows-1,cols-1):return score
        seen.add((r,c))
        for nr,nc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
            if 0<=nr<rows and 0<=nc<cols and (nr,nc) not in seen:
                heappush(pq,(-min(score,grid[nr][nc]),nr,nc))
    raise ValueError('unreachable')


def minimum_cost_cities(n:int,connections:list[list[int]])->int:
    """LC 1135 | Kruskal MST, 1-indexed labels. T O(E log E), S O(n)."""
    if n==1:return 0
    uf=DSU(n+1);total=edges=0
    for a,b,cost in sorted(connections,key=lambda x:x[2]):
        if uf.union(a,b):total+=cost;edges+=1
        if edges==n-1:return total
    return -1


def min_cost_to_supply_water(n:int,wells:list[int],pipes:list[list[int]])->int:
    """LC 1168 | Virtual source node with edges to every well. T O(E log E), S O(n+E)."""
    edges=[[0,i+1,cost] for i,cost in enumerate(wells)]+pipes
    uf=DSU(n+1);cost=0
    for a,b,c in sorted(edges,key=lambda edge:edge[2]):
        if uf.union(a,b):cost+=c
    return cost


def smallest_string_with_swaps(s:str,pairs:list[list[int]])->str:
    """LC 1202 | DSU components then sort each pool. T O(n log n), S O(n)."""
    uf=DSU(len(s))
    for a,b in pairs:uf.union(a,b)
    pools=defaultdict(list)
    for i,char in enumerate(s):pools[uf.find(i)].append(char)
    for component in pools:pools[component].sort(reverse=True)
    return ''.join(pools[uf.find(i)].pop() for i in range(len(s)))


def suggested_products(products:list[str],search_word:str)->list[list[str]]:
    """LC 1268 | Sort + prefix filtering, simple baseline. T O(P log P+P*L), S O(P)."""
    products=sorted(products);result=[];prefix=''
    for char in search_word:
        prefix+=char
        result.append([product for product in products if product.startswith(prefix)][:3])
    return result


def make_connected(n:int,connections:list[list[int]])->int:
    """LC 1319 | DSU count components; extra edges must suffice. T O((n+E)*alpha(n))."""
    if len(connections)<n-1:return -1
    uf=DSU(n)
    for a,b in connections:uf.union(a,b)
    return uf.components-1


def find_city(n:int,edges:list[list[int]],distance_threshold:int)->int:
    """LC 1334 | Floyd-Warshall all-pairs shortest path. T O(n³), S O(n²)."""
    dist=[[inf]*n for _ in range(n)]
    for i in range(n):dist[i][i]=0
    for a,b,w in edges:dist[a][b]=min(dist[a][b],w);dist[b][a]=min(dist[b][a],w)
    for via in range(n):
        for a in range(n):
            for b in range(n):dist[a][b]=min(dist[a][b],dist[a][via]+dist[via][b])
    best_count=inf;best=-1
    for i in range(n):
        count=sum(j!=i and d<=distance_threshold for j,d in enumerate(dist[i]))
        if count<=best_count:best_count=count;best=i
    return best


def constrained_subset_sum(nums:list[int],k:int)->int:
    """LC 1425 | DP + sliding-window max deque. T/S O(n)."""
    if not nums or k<1:raise ValueError('nonempty nums and positive k required')
    dq=deque();dp=[]
    for i,num in enumerate(nums):
        while dq and dq[0]<i-k:dq.popleft()
        best=num+max(0,dp[dq[0]]) if dq else num
        dp.append(best)
        while dq and dp[dq[-1]]<=best:dq.pop()
        dq.append(i)
    return max(dp)


def find_critical_and_pseudo_critical_edges(n:int,edges:list[list[int]])->list[list[int]]:
    """LC 1489 | Kruskal with exclusion/inclusion per edge. T O(E²*alpha(n)), S O(n+E)."""
    ordered=sorted([(w,a,b,i) for i,(a,b,w) in enumerate(edges)])
    def kruskal(exclude=-1,force=-1):
        uf=DSU(n);total=0;used=0
        if force!=-1:
            w,a,b,_=ordered[force]
            if uf.union(a,b):total+=w;used+=1
        for j,(w,a,b,_) in enumerate(ordered):
            if j==exclude or j==force:continue
            if uf.union(a,b):total+=w;used+=1
            if used==n-1:break
        return total if used==n-1 else inf
    baseline=kruskal();critical=[];pseudo=[]
    for j,(_,_,_,index) in enumerate(ordered):
        if kruskal(exclude=j)>baseline:critical.append(index)
        elif kruskal(force=j)==baseline:pseudo.append(index)
    return [sorted(critical),sorted(pseudo)]


def max_value_of_equation(points:list[list[int]],k:int)->int:
    """LC 1499 | Monotone deque of y-x, x sorted. T O(n), S O(n)."""
    dq=deque();best=-inf
    for x,y in points:
        while dq and x-dq[0][0]>k:dq.popleft()
        if dq:best=max(best,x+y+dq[0][1])
        score=y-x
        while dq and dq[-1][1]<=score:dq.pop()
        dq.append((x,score))
    return int(best) if best!=-inf else -1


def max_probability(n:int,edges:list[list[int]],succ_prob:list[float],start:int,end:int)->float:
    """LC 1514 | Maximum-product Dijkstra. T O((V+E) log V), S O(V+E)."""
    graph=defaultdict(list)
    for (a,b),p in zip(edges,succ_prob):graph[a].append((b,p));graph[b].append((a,p))
    best=[0.0]*n;best[start]=1.0;pq=[(-1.0,start)]
    while pq:
        neg,node=heappop(pq);prob=-neg
        if prob<best[node]:continue
        if node==end:return prob
        for other,weight in graph[node]:
            if prob*weight>best[other]:best[other]=prob*weight;heappush(pq,(-best[other],other))
    return 0.0


def max_num_edges_to_remove(n:int,edges:list[list[int]])->int:
    """LC 1579 | Shared edges first, then per-user DSU. T O(E*alpha(n)), S O(n)."""
    alice=DSU(n+1);bob=DSU(n+1);used=0
    for typ,a,b in edges:
        if typ==3:
            if alice.union(a,b):bob.union(a,b);used+=1
    for typ,a,b in edges:
        if typ==1 and alice.union(a,b):used+=1
        elif typ==2 and bob.union(a,b):used+=1
    if len({alice.find(i) for i in range(1,n+1)})!=1:return -1
    if len({bob.find(i) for i in range(1,n+1)})!=1:return -1
    return len(edges)-used


def minimum_effort_path(heights:list[list[int]])->int:
    """LC 1631 | Minimax Dijkstra over adjacent height differences. T O(RC log RC)."""
    rows,cols=len(heights),len(heights[0]);dist=[[inf]*cols for _ in range(rows)]
    dist[0][0]=0;pq=[(0,0,0)]
    while pq:
        effort,r,c=heappop(pq)
        if effort>dist[r][c]:continue
        if (r,c)==(rows-1,cols-1):return effort
        for nr,nc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
            if 0<=nr<rows and 0<=nc<cols:
                new=max(effort,abs(heights[r][c]-heights[nr][nc]))
                if new<dist[nr][nc]:dist[nr][nc]=new;heappush(pq,(new,nr,nc))
    return -1


def max_result_jump_game(nums:list[int],k:int)->int:
    """LC 1696 | Maximum window DP deque. T/S O(n)."""
    if not nums or k<1:raise ValueError('invalid input')
    dp=[nums[0]];dq=deque([0])
    for i in range(1,len(nums)):
        while dq and dq[0]<i-k:dq.popleft()
        dp.append(nums[i]+dp[dq[0]])
        while dq and dp[dq[-1]]<=dp[i]:dq.pop()
        dq.append(i)
    return dp[-1]


def is_covered(ranges:list[list[int]],left:int,right:int)->bool:
    """LC 1893 | Difference sweep over bounded integer values. T O(R+span), S O(span)."""
    if left>right:return True
    diff=defaultdict(int)
    for start,end in ranges:diff[start]+=1;diff[end+1]-=1
    active=0
    for x in range(min([left]+[r[0] for r in ranges]),right+1):
        active+=diff[x]
        if x>=left and active<=0:return False
    return True


def split_painting(segments:list[list[int]])->list[list[int]]:
    """LC 1943 | Sweep color-weight differences. T O(n log n), S O(n)."""
    changes=defaultdict(int)
    for start,end,color in segments:changes[start]+=color;changes[end]-=color
    points=sorted(changes);result=[];total=0
    for i,x in enumerate(points[:-1]):
        total+=changes[x]
        if total and x<points[i+1]:result.append([x,points[i+1],total])
    return result


def count_paths(n:int,roads:list[list[int]])->int:
    """LC 1976 | Dijkstra shortest-path counts mod 1e9+7. T O((V+E)log V)."""
    graph=defaultdict(list)
    for a,b,t in roads:graph[a].append((b,t));graph[b].append((a,t))
    mod=10**9+7;dist=[inf]*n;ways=[0]*n
    dist[0]=0;ways[0]=1;pq=[(0,0)]
    while pq:
        d,u=heappop(pq)
        if d!=dist[u]:continue
        for v,w in graph[u]:
            new=d+w
            if new<dist[v]:dist[v]=new;ways[v]=ways[u];heappush(pq,(new,v))
            elif new==dist[v]:ways[v]=(ways[v]+ways[u])%mod
    return ways[-1]


def shifting_letters(s:str,shifts:list[list[int]])->str:
    """LC 2381 | Difference array of +1/-1 alphabet shifts. T O(n+Q), S O(n)."""
    diff=[0]*(len(s)+1)
    for start,end,direction in shifts:
        delta=1 if direction else -1
        diff[start]+=delta;diff[end+1]-=delta
    run=0;result=[]
    for i,ch in enumerate(s):
        run+=diff[i];result.append(chr((ord(ch)-97+run)%26+97))
    return ''.join(result)
