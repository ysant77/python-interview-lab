"""Additional high-value interview solutions, with LeetCode IDs and complexity.

The implementations are independent reference solutions written for this lab.
"""
from __future__ import annotations
from collections import Counter, defaultdict, deque
from heapq import heappop, heappush
from math import ceil
from typing import Optional
from .structures import TreeNode


def permute_unique(nums: list[int]) -> list[list[int]]:
    """LC 47 | Frequency-map backtracking. T O(n*n!) worst, auxiliary S O(n)."""
    counter = Counter(nums); result = []; cur = []
    def visit():
        if len(cur) == len(nums): result.append(cur.copy()); return
        for number in counter:
            if counter[number]:
                counter[number] -= 1; cur.append(number)
                visit()
                cur.pop(); counter[number] += 1
    visit(); return result


class NextNode:
    """Binary tree node with next pointer (LC 116)."""
    def __init__(self,val: int = 0,left: Optional['NextNode'] = None,
                 right: Optional['NextNode'] = None,next: Optional['NextNode'] = None):
        self.val=val; self.left=left; self.right=right; self.next=next


def connect_next(root: Optional[NextNode]) -> Optional[NextNode]:
    """LC 116 | BFS linking across levels. T O(n), S O(width)."""
    if root is None:return None
    queue=deque([root])
    while queue:
        prev=None
        for _ in range(len(queue)):
            node=queue.popleft()
            if prev is not None:prev.next=node
            prev=node
            if node.left:queue.append(node.left)
            if node.right:queue.append(node.right)
        if prev:prev.next=None
    return root


def gas_station(gas:list[int],cost:list[int])->int:
    """LC 134 | Greedy reset after negative running balance. T O(n), S O(1)."""
    if len(gas)!=len(cost) or not gas:return -1
    if sum(gas)<sum(cost):return -1
    start=0;balance=0
    for i,(g,c) in enumerate(zip(gas,cost)):
        balance+=g-c
        if balance<0:start=i+1;balance=0
    return start


class RandomNode:
    """Linked-list node with arbitrary random pointer."""
    def __init__(self,val:int,next:Optional['RandomNode']=None,random:Optional['RandomNode']=None):
        self.val=val;self.next=next;self.random=random


def copy_random_list(head:Optional[RandomNode])->Optional[RandomNode]:
    """LC 138 | Old-node to new-node hashmap. T O(n), S O(n)."""
    old_to_new={None:None}; cur=head
    while cur:
        old_to_new[cur]=RandomNode(cur.val);cur=cur.next
    cur=head
    while cur:
        old_to_new[cur].next=old_to_new[cur.next]
        old_to_new[cur].random=old_to_new[cur.random]
        cur=cur.next
    return old_to_new[head]


def reconstruct_queue(people:list[list[int]])->list[list[int]]:
    """LC 406 | Tallest-first insertion. T O(n^2), S O(n) output."""
    result=[]
    for person in sorted(people,key=lambda x:(-x[0],x[1])):
        result.insert(person[1],person[:])
    return result


def split_array_largest_sum(nums:list[int],k:int)->int:
    """LC 410 | Binary search feasible maximum group sum. T O(n log(sum)), S O(1)."""
    if not nums or any(x<0 for x in nums) or not 1<=k<=len(nums):
        raise ValueError('nonnegative array and 1<=k<=n required')
    lo,hi=max(nums),sum(nums)
    while lo<hi:
        mid=(lo+hi)//2;groups=1;running=0
        for n in nums:
            if running+n>mid:groups+=1;running=n
            else:running+=n
        if groups<=k:hi=mid
        else:lo=mid+1
    return lo


def largest_values(root:Optional[TreeNode])->list[int]:
    """LC 515 | Level-order maximum. T O(n), S O(width)."""
    if root is None:return []
    q=deque([root]);result=[]
    while q:
        maximum=float('-inf')
        for _ in range(len(q)):
            node=q.popleft();maximum=max(maximum,node.val)
            if node.left:q.append(node.left)
            if node.right:q.append(node.right)
        result.append(int(maximum))
    return result


def check_subarray_sum(nums:list[int],k:int)->bool:
    """LC 523 | Matching prefix remainders >=2 indices apart. T O(n), S O(n)."""
    earliest={0:-1};running=0
    for i,n in enumerate(nums):
        running+=n;rem=running%k if k else running
        if rem in earliest:
            if i-earliest[rem]>=2:return True
        else:earliest[rem]=i
    return False


def least_interval(tasks:list[str],n:int)->int:
    """LC 621 | Most frequent task determines idle slots. T O(m), S O(alphabet)."""
    if n<0:raise ValueError('nonnegative cooldown required')
    if not tasks:return 0
    freq=Counter(tasks);highest=max(freq.values());ties=sum(v==highest for v in freq.values())
    return max(len(tasks),(highest-1)*(n+1)+ties)


def average_of_levels(root:Optional[TreeNode])->list[float]:
    """LC 637 | BFS per level. T O(n), S O(width)."""
    if root is None:return []
    q=deque([root]);result=[]
    while q:
        count=len(q);total=0
        for _ in range(count):
            node=q.popleft();total+=node.val
            if node.left:q.append(node.left)
            if node.right:q.append(node.right)
        result.append(total/count)
    return result


def minmax_gas_dist(stations:list[int],k:int,eps:float=1e-6)->float:
    """LC 774 | Float binary search on max gap. T O(n*log(range/eps)), S O(1)."""
    if len(stations)<2 or k<0 or eps<=0:raise ValueError('invalid stations/k/eps')
    station=sorted(stations);gaps=[b-a for a,b in zip(station,station[1:])]
    if not gaps or max(gaps)==0:return 0.0
    lo=0.0;hi=float(max(gaps))
    while hi-lo>eps:
        mid=(lo+hi)/2
        # Exact multiple gap/mid needs one fewer station, tolerate FP wobble.
        needed=sum(max(0,ceil(gap/mid-1e-12)-1) for gap in gaps)
        if needed<=k:hi=mid
        else:lo=mid
    return hi


def eventual_safe_nodes(graph:list[list[int]])->list[int]:
    """LC 802 | Reverse Kahn peel of terminal nodes. T O(V+E), S O(V+E)."""
    indegree=[len(outgoing) for outgoing in graph]
    reverse=[[] for _ in graph]
    for u,edges in enumerate(graph):
        for v in edges:reverse[v].append(u)
    q=deque(i for i,d in enumerate(indegree) if d==0)
    while q:
        node=q.popleft()
        for before in reverse[node]:
            indegree[before]-=1
            if indegree[before]==0:q.append(before)
    return [i for i,remaining in enumerate(indegree) if remaining==0]


def num_subarrays_with_sum(nums:list[int],goal:int)->int:
    """LC 930 | Prefix sum counts, works for arbitrary integers. T O(n), S O(n)."""
    count=defaultdict(int);count[0]=1;s=ans=0
    for num in nums:
        s+=num;ans+=count[s-goal];count[s]+=1
    return ans


def subarrays_div_by_k(nums:list[int],k:int)->int:
    """LC 974 | Count equal prefix residues. T O(n), S O(min(n,|k|))."""
    if k==0:raise ValueError('k must be nonzero')
    counts=defaultdict(int);counts[0]=1;prefix=ans=0
    for num in nums:
        prefix=(prefix+num)%k;ans+=counts[prefix];counts[prefix]+=1
    return ans


def smallest_divisor(nums:list[int],threshold:int)->int:
    """LC 1283 | Binary search divisor. T O(n log(max(nums))), S O(1)."""
    if not nums or any(x<=0 for x in nums) or threshold<len(nums):
        raise ValueError('positive array and threshold >= n required')
    lo,hi=1,max(nums)
    while lo<hi:
        mid=(lo+hi)//2
        if sum((x+mid-1)//mid for x in nums)<=threshold:hi=mid
        else:lo=mid+1
    return lo


def min_days_bouquets(bloom_day:list[int],m:int,k:int)->int:
    """LC 1482 | Binary search day + consecutive-run check. T O(n log R), S O(1)."""
    if m==0:return 0
    if not bloom_day or m*k>len(bloom_day) or m<0 or k<1:return -1
    lo,hi=min(bloom_day),max(bloom_day)
    def possible(day):
        bouquets=run=0
        for bloom in bloom_day:
            run=run+1 if bloom<=day else 0
            if run==k:bouquets+=1;run=0
        return bouquets>=m
    while lo<hi:
        mid=(lo+hi)//2
        if possible(mid):hi=mid
        else:lo=mid+1
    return lo


def check_if_prerequisite(n:int,prerequisites:list[list[int]],queries:list[list[int]])->list[bool]:
    """LC 1462 | Transitive reachability via Floyd-Warshall. T O(n^3+Q), S O(n^2)."""
    reach=[[False]*n for _ in range(n)]
    for a,b in prerequisites:reach[a][b]=True
    for mid in range(n):
        for src in range(n):
            if reach[src][mid]:
                for dst in range(n):reach[src][dst]|=reach[mid][dst]
    return [reach[a][b] for a,b in queries]


def find_all_recipes(recipes:list[str],ingredients:list[list[str]],supplies:list[str])->list[str]:
    """LC 2115 | Topological production when ingredients available. T O(V+E), S O(V+E)."""
    pending={recipe:len(parts) for recipe,parts in zip(recipes,ingredients)}
    dependents=defaultdict(list)
    for recipe,parts in zip(recipes,ingredients):
        for part in parts:dependents[part].append(recipe)
    q=deque(supplies);result=[]
    for recipe,n in pending.items():
        if n==0:q.append(recipe)
    emitted=set()
    while q:
        item=q.popleft()
        if item in pending and pending[item]==0 and item not in emitted:
            result.append(item);emitted.add(item)
        for recipe in dependents[item]:
            pending[recipe]-=1
            if pending[recipe]==0:q.append(recipe)
    return result


def sort_items_by_groups(n:int,m:int,group:list[int],before_items:list[list[int]])->list[int]:
    """LC 1203 | Two topological sorts (items and groups). T O(n+m+E), S O(n+m+E)."""
    if len(group)!=n or len(before_items)!=n:raise ValueError('invalid input lengths')
    group=group[:]
    for i in range(n):
        if group[i]==-1:group[i]=m;m+=1
    item_graph=[[] for _ in range(n)];item_deg=[0]*n
    group_graph=[set() for _ in range(m)];group_deg=[0]*m
    for item,reqs in enumerate(before_items):
        for predecessor in reqs:
            item_graph[predecessor].append(item);item_deg[item]+=1
            a,b=group[predecessor],group[item]
            if a!=b and b not in group_graph[a]:
                group_graph[a].add(b);group_deg[b]+=1
    def topo(edges,degrees):
        queue=deque(i for i,v in enumerate(degrees) if v==0);res=[]
        while queue:
            v=queue.popleft();res.append(v)
            for other in edges[v]:
                degrees[other]-=1
                if degrees[other]==0:queue.append(other)
        return res if len(res)==len(edges) else []
    item_order=topo(item_graph,item_deg)
    group_order=topo(group_graph,group_deg)
    if len(item_order)!=n or len(group_order)!=m:return []
    by_group=defaultdict(list)
    for item in item_order:by_group[group[item]].append(item)
    return [item for grp in group_order for item in by_group[grp]]
