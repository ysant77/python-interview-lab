"""Further solutions linked by the supplementary public DSA-patterns reference.

LeetCode IDs are recorded solely to locate the original problem statements.
These are newly written Python solutions, not transcriptions of source code.
"""
from __future__ import annotations
from collections import Counter, defaultdict, deque
from functools import lru_cache
from heapq import heappop, heappush
from math import inf
from typing import Optional
from .structures import ListNode, TreeNode


def remove_element(nums:list[int],val:int)->int:
    """LC 27 | Compact in place. T O(n), S O(1)."""
    write=0
    for x in nums:
        if x!=val:nums[write]=x;write+=1
    return write


def rotate_right(head:Optional[ListNode],k:int)->Optional[ListNode]:
    """LC 61 | Make a ring, break at new tail. T O(n), S O(1)."""
    if head is None or head.next is None or k==0:return head
    length=1;tail=head
    while tail.next:tail=tail.next;length+=1
    k%=length
    if k==0:return head
    tail.next=head
    for _ in range(length-k):tail=tail.next
    new_head=tail.next;tail.next=None
    return new_head


def search_rotated_ii(nums:list[int],target:int)->bool:
    """LC 81 | Binary search with duplicate ambiguity. Average O(log n), worst O(n)."""
    lo,hi=0,len(nums)-1
    while lo<=hi:
        mid=(lo+hi)//2
        if nums[mid]==target:return True
        if nums[lo]==nums[mid]==nums[hi]:lo+=1;hi-=1
        elif nums[lo]<=nums[mid]:
            if nums[lo]<=target<nums[mid]:hi=mid-1
            else:lo=mid+1
        else:
            if nums[mid]<target<=nums[hi]:lo=mid+1
            else:hi=mid-1
    return False


def unique_bst_trees(n:int)->list[Optional[TreeNode]]:
    """LC 95 | Recursively enumerate structurally unique BSTs. Output O(C_n*n)."""
    if n<0:raise ValueError('nonnegative n required')
    if n==0:return []
    def generate(lo,hi):
        if lo>hi:return [None]
        result=[]
        for root in range(lo,hi+1):
            for left in generate(lo,root-1):
                for right in generate(root+1,hi):
                    # Clone to avoid sharing mutable subtree nodes across outputs.
                    def clone(node):
                        return TreeNode(node.val,clone(node.left),clone(node.right)) if node else None
                    result.append(TreeNode(root,clone(left),clone(right)))
        return result
    return generate(1,n)


def num_unique_bsts(n:int)->int:
    """LC 96 | Catalan DP. T O(n²), S O(n)."""
    if n<0:raise ValueError('nonnegative n required')
    dp=[0]*(n+1);dp[0]=1
    for nodes in range(1,n+1):
        dp[nodes]=sum(dp[left]*dp[nodes-1-left] for left in range(nodes))
    return dp[n]


def level_order_bottom(root:Optional[TreeNode])->list[list[int]]:
    """LC 107 | BFS levels then reverse. T O(n), S O(n)."""
    if root is None:return []
    queue=deque([root]);answer=[]
    while queue:
        level=[]
        for _ in range(len(queue)):
            node=queue.popleft();level.append(node.val)
            if node.left:queue.append(node.left)
            if node.right:queue.append(node.right)
        answer.append(level)
    return answer[::-1]


def min_depth(root:Optional[TreeNode])->int:
    """LC 111 | BFS until first leaf. T O(n), S O(width)."""
    if root is None:return 0
    queue=deque([(root,1)])
    while queue:
        node,depth=queue.popleft()
        if node.left is None and node.right is None:return depth
        if node.left:queue.append((node.left,depth+1))
        if node.right:queue.append((node.right,depth+1))
    return 0


def sum_numbers(root:Optional[TreeNode])->int:
    """LC 129 | Root-to-leaf decimal accumulation. T O(n), S O(height)."""
    def dfs(node,prefix):
        if node is None:return 0
        value=prefix*10+node.val
        if node.left is None and node.right is None:return value
        return dfs(node.left,value)+dfs(node.right,value)
    return dfs(root,0)


def reorder_list(head:Optional[ListNode])->None:
    """LC 143 | Split, reverse second half, merge alternating. T O(n), S O(1)."""
    if head is None or head.next is None:return
    slow=fast=head
    while fast.next and fast.next.next:slow=slow.next;fast=fast.next.next
    second=slow.next;slow.next=None;prev=None
    while second:next_node=second.next;second.next=prev;prev=second;second=next_node
    first=head;second=prev
    while second:
        first_next,second_next=first.next,second.next
        first.next=second;second.next=first_next
        first,second=first_next,second_next


def find_min_rotated_ii(nums:list[int])->int:
    """LC 154 | Shrink ambiguous right bound. Average O(log n), worst O(n)."""
    if not nums:raise ValueError('nonempty array required')
    lo,hi=0,len(nums)-1
    while lo<hi:
        mid=(lo+hi)//2
        if nums[mid]<nums[hi]:hi=mid
        elif nums[mid]>nums[hi]:lo=mid+1
        else:hi-=1
    return nums[lo]


def is_happy(n:int)->bool:
    """LC 202 | Repeated sum-of-squares cycle detection. O(log n) per step."""
    if n<=0:return False
    seen=set()
    while n!=1 and n not in seen:
        seen.add(n);n=sum(int(c)**2 for c in str(n))
    return n==1


def diff_ways_to_compute(expression:str)->list[int]:
    """LC 241 | Divide at each operator with memoization; exponential outputs."""
    @lru_cache(None)
    def solve(s):
        answer=[]
        for i,ch in enumerate(s):
            if ch in '+-*':
                for left in solve(s[:i]):
                    for right in solve(s[i+1:]):
                        answer.append(left+right if ch=='+' else left-right if ch=='-' else left*right)
        return answer if answer else [int(s)]
    return solve(expression)


def binary_tree_paths(root:Optional[TreeNode])->list[str]:
    """LC 257 | DFS building root-to-leaf strings. T O(total output chars), S O(height)."""
    result=[]
    def dfs(node,path):
        if node is None:return
        here=path+[str(node.val)]
        if node.left is None and node.right is None:result.append('->'.join(here))
        else:dfs(node.left,here);dfs(node.right,here)
    dfs(root,[]);return result


def three_sum_smaller(nums:list[int],target:int)->int:
    """LC 259 | Sorted outer anchor + inner two pointers. T O(n²), S O(1) apart sort."""
    nums.sort();count=0
    for i in range(len(nums)-2):
        lo,hi=i+1,len(nums)-1
        while lo<hi:
            if nums[i]+nums[lo]+nums[hi]<target:count+=hi-lo;lo+=1
            else:hi-=1
    return count


def alien_order(words:list[str])->str:
    """LC 269 | Precedence DAG, Kahn topological ordering. T O(total chars + E + V)."""
    graph={char:set() for word in words for char in word};degree={c:0 for c in graph}
    for first,second in zip(words,words[1:]):
        if len(first)>len(second) and first.startswith(second):return ''
        for a,b in zip(first,second):
            if a!=b:
                if b not in graph[a]:graph[a].add(b);degree[b]+=1
                break
    queue=deque(sorted(c for c,d in degree.items() if d==0));result=[]
    while queue:
        char=queue.popleft();result.append(char)
        for other in sorted(graph[char]):
            degree[other]-=1
            if degree[other]==0:queue.append(other)
    return ''.join(result) if len(result)==len(graph) else ''


def find_duplicate(nums:list[int])->int:
    """LC 287 | Floyd cycle in index functional graph. T O(n), S O(1)."""
    slow=fast=nums[0]
    while True:
        slow=nums[slow];fast=nums[nums[fast]]
        if slow==fast:break
    slow=nums[0]
    while slow!=fast:slow=nums[slow];fast=nums[fast]
    return slow


def generate_abbreviations(word:str)->list[str]:
    """LC 320 | DFS abbreviation count with numeric runs. T O(n*2^n), S O(n) aux."""
    result=[]
    def dfs(i,count,out):
        if i==len(word):result.append(out+(str(count) if count else ''));return
        dfs(i+1,count+1,out)
        dfs(i+1,0,out+(str(count) if count else '')+word[i])
    dfs(0,0,'');return result


def path_sum_iii(root:Optional[TreeNode],target_sum:int)->int:
    """LC 437 | Prefix-sum hash counts along root path. T O(n), S O(height)."""
    counts={0:1}
    def dfs(node,prefix):
        if node is None:return 0
        prefix+=node.val;answer=counts.get(prefix-target_sum,0)
        counts[prefix]=counts.get(prefix,0)+1
        answer+=dfs(node.left,prefix)+dfs(node.right,prefix)
        counts[prefix]-=1
        return answer
    return dfs(root,0)


def find_duplicates(nums:list[int])->list[int]:
    """LC 442 | In-place negative marking, values 1..n. T O(n), S O(1) aux."""
    result=[]
    for value in nums:
        index=abs(value)-1
        if nums[index]<0:result.append(abs(value))
        else:nums[index]=-nums[index]
    return result


def sequence_reconstruction(nums:list[int],sequences:list[list[int]])->bool:
    """LC 444 | Unique topo order must equal nums. T O(V+E), S O(V+E)."""
    nodes=set(nums);g={x:set() for x in nums};degree={x:0 for x in nums};observed=set()
    for seq in sequences:
        for item in seq:
            if item not in nodes:return False
            observed.add(item)
        for before,after in zip(seq,seq[1:]):
            if after not in g[before]:g[before].add(after);degree[after]+=1
    if observed!=nodes:return False
    queue=deque(x for x,d in degree.items() if d==0)
    for expected in nums:
        if len(queue)!=1 or queue[0]!=expected:return False
        item=queue.popleft()
        for after in g[item]:
            degree[after]-=1
            if degree[after]==0:queue.append(after)
    return not queue


def circular_array_loop(nums:list[int])->bool:
    """LC 457 | Direction-consistent Floyd pointers. T O(n²) simple variant, S O(1)."""
    n=len(nums)
    if n==0:return False
    def following(i,forward):
        if (nums[i]>0)!=forward:return -1
        j=(i+nums[i])%n
        return -1 if j==i or (nums[j]>0)!=forward else j
    for start in range(n):
        slow=fast=start;forward=nums[start]>0
        while True:
            slow=following(slow,forward) if slow!=-1 else -1
            fast=following(fast,forward) if fast!=-1 else -1
            fast=following(fast,forward) if fast!=-1 else -1
            if slow==-1 or fast==-1:break
            if slow==fast:return True
    return False


def find_maximized_capital(k:int,w:int,profits:list[int],capital:list[int])->int:
    """LC 502 | Capital-sorted projects + profit maxheap. T O(n log n + k log n), S O(n)."""
    projects=sorted(zip(capital,profits));pq=[];i=0
    for _ in range(k):
        while i<len(projects) and projects[i][0]<=w:
            heappush(pq,-projects[i][1]);i+=1
        if not pq:break
        best=-heappop(pq)
        if best<=0:break
        w+=best
    return w


def search_unknown_size(reader,target:int)->int:
    """LC 702 | Exponential boundary expansion then binary search. O(log target_index)."""
    def get(i):return reader.get(i)
    lo,hi=0,1
    while get(hi)<target:lo=hi+1;hi=hi*2+1
    while lo<=hi:
        mid=(lo+hi)//2;value=get(mid)
        if value==target:return mid
        if value<target:lo=mid+1
        else:hi=mid-1
    return -1


def num_subarray_product_less_than_k(nums:list[int],k:int)->int:
    """LC 713 | Positive nums sliding product. T O(n), S O(1)."""
    if k<=1:return 0
    product=1;left=0;answer=0
    for right,x in enumerate(nums):
        product*=x
        while product>=k:product//=nums[left];left+=1
        answer+=right-left+1
    return answer


def next_greatest_letter(letters:list[str],target:str)->str:
    """LC 744 | Upper-bound search with wraparound. T O(log n), S O(1)."""
    lo,hi=0,len(letters)
    while lo<hi:
        mid=(lo+hi)//2
        if letters[mid]<=target:lo=mid+1
        else:hi=mid
    return letters[lo%len(letters)]


def employee_free_time(schedule:list[list[list[int]]])->list[list[int]]:
    """LC 759 | Merge busy intervals, return finite gaps. T O(n log n), S O(n)."""
    intervals=sorted((a,b) for employee in schedule for a,b in employee)
    if not intervals:return []
    end=intervals[0][1];free=[]
    for start,stop in intervals[1:]:
        if start>end:free.append([end,start])
        end=max(end,stop)
    return free


def flip_and_invert_image(image:list[list[int]])->list[list[int]]:
    """LC 832 | Reverse each binary row and invert bits. T O(RC), S O(RC) output."""
    return [[1^bit for bit in row[::-1]] for row in image]


def backspace_compare(s:str,t:str)->bool:
    """LC 844 | Reverse traversal with skipped characters. T O(n+m), S O(1)."""
    def rev(text):
        skip=0
        for ch in reversed(text):
            if ch=='#':skip+=1
            elif skip:skip-=1
            else:yield ch
    from itertools import zip_longest
    return all(a==b for a,b in zip_longest(rev(s),rev(t),fillvalue=None))


def interval_intersection(first:list[list[int]],second:list[list[int]])->list[list[int]]:
    """LC 986 | Sorted interval two pointers. T O(n+m), S O(output)."""
    i=j=0;result=[]
    while i<len(first) and j<len(second):
        lo=max(first[i][0],second[j][0]);hi=min(first[i][1],second[j][1])
        if lo<=hi:result.append([lo,hi])
        if first[i][1]<second[j][1]:i+=1
        else:j+=1
    return result


def bitwise_complement(n:int)->int:
    """LC 1009 | Flip significant bits; 0 has 1-bit representation. T/S O(1)."""
    return n ^ ((1 << max(1,n.bit_length()))-1)


def max_level_sum(root:Optional[TreeNode])->int:
    """LC 1161 | BFS sums; smallest level on ties. T O(n), S O(width)."""
    if not root:return 0
    q=deque([root]);best=-inf;result=1;level=0
    while q:
        level+=1;total=0
        for _ in range(len(q)):
            node=q.popleft();total+=node.val
            if node.left:q.append(node.left)
            if node.right:q.append(node.right)
        if total>best:best=total;result=level
    return result


def connect_sticks(sticks:list[int])->int:
    """LC 1167 | Greedy minheap merging two smallest. T O(n log n), S O(n)."""
    heap=sticks.copy()
    from heapq import heapify
    heapify(heap);cost=0
    while len(heap)>1:
        merged=heappop(heap)+heappop(heap)
        cost+=merged;heappush(heap,merged)
    return cost


def minimum_abs_difference(arr:list[int])->list[list[int]]:
    """LC 1200 | Sort, compare adjacent differences. T O(n log n), S O(output)."""
    arr=sorted(arr)
    if len(arr)<2:return []
    best=min(b-a for a,b in zip(arr,arr[1:]))
    return [[a,b] for a,b in zip(arr,arr[1:]) if b-a==best]


def find_kth_positive(arr:list[int],k:int)->int:
    """LC 1539 | Missing count at i = arr[i]-i-1. T O(log n), S O(1)."""
    lo,hi=0,len(arr)
    while lo<hi:
        mid=(lo+hi)//2
        if arr[mid]-mid-1<k:lo=mid+1
        else:hi=mid
    return lo+k


def min_cost_cut_stick(n:int,cuts:list[int])->int:
    """LC 1547 | Interval DP between sorted cut positions. T O(C³), S O(C²)."""
    points=[0]+sorted(cuts)+[n];m=len(points)
    dp=[[0]*m for _ in range(m)]
    for length in range(2,m):
        for i in range(m-length):
            j=i+length
            dp[i][j]=min((dp[i][k]+dp[k][j]+points[j]-points[i] for k in range(i+1,j)),default=0)
    return dp[0][-1]


def find_length_of_shortest_subarray(arr:list[int])->int:
    """LC 1574 | Nondecreasing prefix/suffix merge. T O(n), S O(1)."""
    n=len(arr)
    if n<2:return 0
    right=n-1
    while right>0 and arr[right-1]<=arr[right]:right-=1
    if right==0:return 0
    answer=right;left=0
    while left==0 or (left<n and arr[left]>=arr[left-1]):
        while right<n and arr[left]>arr[right]:right+=1
        answer=min(answer,right-left-1)
        left+=1
        if left>=n:break
    return answer
