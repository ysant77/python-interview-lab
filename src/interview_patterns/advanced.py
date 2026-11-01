"""Additional solutions inspired by well-known patterns (original Python code)."""
from __future__ import annotations
from collections import defaultdict,deque
from heapq import heappop,heappush
from itertools import count
from .structures import ListNode


def merge_k_lists(lists:list[ListNode|None])->ListNode|None:
    """LC 23. Heap with tie-breaker. O(N log k), O(k) heap."""
    heap=[];tie=count()
    for node in lists:
        if node:heappush(heap,(node.val,next(tie),node))
    dummy=tail=ListNode()
    while heap:
        _,_,node=heappop(heap);tail.next=node;tail=node
        if node.next:heappush(heap,(node.next.val,next(tie),node.next))
    if tail:tail.next=None
    return dummy.next


def kth_smallest_sorted_matrix(matrix:list[list[int]],k:int)->int:
    """LC 378. Heap merge sorted rows. O(k log min(k,R)), O(R) heap."""
    if not matrix or not matrix[0] or not 1<=k<=sum(map(len,matrix)):raise ValueError('invalid k')
    h=[]
    for r,row in enumerate(matrix):
        if row:heappush(h,(row[0],r,0))
    for _ in range(k):
        val,r,c=heappop(h)
        if c+1<len(matrix[r]):heappush(h,(matrix[r][c+1],r,c+1))
    return val


def smallest_range(lists:list[list[int]])->list[int]:
    """LC 632. Heap over k sorted nonempty lists. O(N log k), O(k)."""
    if not lists or any(not x for x in lists):raise ValueError('nonempty sorted lists needed')
    h=[];hi=float('-inf')
    for i,row in enumerate(lists):heappush(h,(row[0],i,0));hi=max(hi,row[0])
    best=[h[0][0],hi]
    while True:
        low,r,c=heappop(h)
        if hi-low < best[1]-best[0] or (hi-low==best[1]-best[0] and low<best[0]):best=[low,hi]
        if c+1==len(lists[r]):return best
        val=lists[r][c+1];hi=max(hi,val);heappush(h,(val,r,c+1))


def find_k_pairs_smallest(nums1:list[int],nums2:list[int],k:int)->list[list[int]]:
    """LC 373. K-way sums from sorted arrays. O(k log min(k,m)), O(k)."""
    if not nums1 or not nums2 or k<=0:return []
    heap=[(nums1[i]+nums2[0],i,0) for i in range(min(k,len(nums1)))]
    from heapq import heapify
    heapify(heap);out=[]
    while heap and len(out)<k:
        _,i,j=heappop(heap);out.append([nums1[i],nums2[j]])
        if j+1<len(nums2):heappush(heap,(nums1[i]+nums2[j+1],i,j+1))
    return out


def median_sliding_window(nums:list[int],k:int)->list[float]:
    """LC 480. Sorted window via bisect. O(n*k) insert/delete, O(k) space.

    Teaching baseline; two lazy-deletion heaps can improve to O(n log k).
    """
    from bisect import bisect_left,insort
    if not 1<=k<=len(nums):return []
    window=sorted(nums[:k]);out=[]
    def median():return float(window[k//2]) if k%2 else (window[k//2-1]+window[k//2])/2
    out.append(median())
    for i in range(k,len(nums)):
        window.pop(bisect_left(window,nums[i-k]));insort(window,nums[i]);out.append(median())
    return out


def find_right_interval(intervals:list[list[int]])->list[int]:
    """LC 436. Binary search smallest start >= each end. O(n log n), O(n)."""
    from bisect import bisect_left
    ordered=sorted((start,i) for i,(start,end) in enumerate(intervals));starts=[x for x,_ in ordered]
    out=[]
    for _,end in intervals:
        j=bisect_left(starts,end);out.append(ordered[j][1] if j<len(ordered) else -1)
    return out


def single_number_iii(nums:list[int])->list[int]:
    """LC 260. Split by one differing set bit. O(n), O(1)."""
    xor=0
    for x in nums:xor^=x
    bit=xor & -xor;a=b=0
    for x in nums:
        if x&bit:a^=x
        else:b^=x
    return [a,b]


def generate_parenthesis(n:int)->list[str]:
    """LC 22. Never close more parentheses than opened. Catalan output."""
    out=[]
    def dfs(s,opens,closes):
        if len(s)==2*n:out.append(s);return
        if opens<n:dfs(s+'(',opens+1,closes)
        if closes<opens:dfs(s+')',opens,closes+1)
    dfs('',0,0);return out


def letter_case_permutation(s:str)->list[str]:
    """LC 784. Branch only on letters. O(n*2^letters) time/output."""
    out=['']
    for c in s:
        if c.isalpha():out=[p+c.lower() for p in out]+[p+c.upper() for p in out]
        else:out=[p+c for p in out]
    return out


def four_sum(nums:list[int],target:int)->list[list[int]]:
    """LC 18. Sorted two pointers inside two loops. O(n^3), O(n) sort space."""
    nums=sorted(nums);n=len(nums);ans=[]
    for i in range(n-3):
        if i and nums[i]==nums[i-1]:continue
        for j in range(i+1,n-2):
            if j>i+1 and nums[j]==nums[j-1]:continue
            l,r=j+1,n-1
            while l<r:
                val=nums[i]+nums[j]+nums[l]+nums[r]
                if val<target:l+=1
                elif val>target:r-=1
                else:
                    ans.append([nums[i],nums[j],nums[l],nums[r]])
                    l+=1;r-=1
                    while l<r and nums[l]==nums[l-1]:l+=1
    return ans


def palindrome_partition(s:str)->list[list[str]]:
    """LC 131. Backtrack across palindromic prefixes. O(n*2^n) time/output."""
    out=[];path=[]
    def visit(start):
        if start==len(s):out.append(path[:]);return
        for end in range(start+1,len(s)+1):
            p=s[start:end]
            if p==p[::-1]:path.append(p);visit(end);path.pop()
    visit(0);return out


def combination_sum_ii(candidates:list[int],target:int)->list[list[int]]:
    """LC 40. Each index at most once; skip duplicates at same depth. Exponential output."""
    candidates=sorted(candidates);out=[];path=[]
    def visit(start,remaining):
        if remaining==0:out.append(path[:]);return
        for i in range(start,len(candidates)):
            if i>start and candidates[i]==candidates[i-1]:continue
            x=candidates[i]
            if x>remaining:break
            path.append(x);visit(i+1,remaining-x);path.pop()
    visit(0,target);return out


def solve_n_queens(n:int)->list[list[str]]:
    """LC 51. Row-by-row columns and diagonals. O(n!) search, O(n) aux + output."""
    out=[];cols=set();down=set();up=set();placements=[]
    def visit(r):
        if r==n:
            out.append(['.'*c+'Q'+'.'*(n-c-1) for c in placements]);return
        for c in range(n):
            if c in cols or r-c in down or r+c in up:continue
            cols.add(c);down.add(r-c);up.add(r+c);placements.append(c)
            visit(r+1)
            placements.pop();cols.remove(c);down.remove(r-c);up.remove(r+c)
    visit(0);return out


def min_height_trees(n:int,edges:list[list[int]])->list[int]:
    """LC 310. Peel leaves; tree input. O(V+E), O(V+E)."""
    if n==1:return [0]
    adj=[set() for _ in range(n)]
    for a,b in edges:adj[a].add(b);adj[b].add(a)
    leaves=deque(i for i in range(n) if len(adj[i])==1);left=n
    while left>2:
        count=len(leaves);left-=count
        for _ in range(count):
            leaf=leaves.popleft();neighbor=next(iter(adj[leaf]));adj[neighbor].remove(leaf)
            if len(adj[neighbor])==1:leaves.append(neighbor)
    return list(leaves)
