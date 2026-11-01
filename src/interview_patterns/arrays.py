"""Original Python reference solutions for array, string and search patterns.

Complexities describe auxiliary space, excluding returned output unless stated.
Prefer reasoning about the loop invariant before memorizing an implementation.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict, deque
from heapq import heappush, heappop, nlargest
from typing import Sequence


def two_sum(nums: list[int], target: int) -> list[int]:
    """LC 1. Store numbers seen; find complement. O(n) time/space."""
    seen = {}
    for i, x in enumerate(nums):
        if target - x in seen:
            return [seen[target - x], i]
        seen[x] = i
    return []


def valid_anagram(s: str, t: str) -> bool:
    """LC 242. Count code points. O(n+m) time, O(k) space."""
    return Counter(s) == Counter(t)


def contains_duplicate(nums: list[int]) -> bool:
    """LC 217. Hash set. O(n) expected time, O(n) space."""
    return len(nums) != len(set(nums))


def group_anagrams(words: list[str]) -> list[list[str]]:
    """LC 49. Sorted-character signature. O(N K log K) time, O(NK) space."""
    buckets = defaultdict(list)
    for w in words:
        buckets[tuple(sorted(w))].append(w)
    return list(buckets.values())


def longest_consecutive(nums: list[int]) -> int:
    """LC 128. Grow only from sequence starts. Expected O(n) time/space."""
    s = set(nums); best = 0
    for x in s:
        if x - 1 not in s:
            y = x
            while y in s:
                y += 1
            best = max(best, y - x)
    return best


def longest_unique_substring(s: str) -> int:
    """LC 3. No duplicate in [left, right]. O(n) time, O(k) space."""
    last = {}; left = 0; answer = 0
    for right, ch in enumerate(s):
        if ch in last:
            left = max(left, last[ch] + 1)
        last[ch] = right
        answer = max(answer, right - left + 1)
    return answer


def min_subarray_len(target: int, nums: list[int]) -> int:
    """LC 209. Positive numbers only. O(n) time, O(1) extra space."""
    left = total = 0; best = len(nums) + 1
    for right, x in enumerate(nums):
        total += x
        while total >= target:
            best = min(best, right - left + 1)
            total -= nums[left]; left += 1
    return 0 if best > len(nums) else best


def character_replacement(s: str, k: int) -> int:
    """LC 424. Sliding window with monotonic max frequency. O(n) time, O(k) space."""
    count = defaultdict(int); left = max_seen = best = 0
    for right, ch in enumerate(s):
        count[ch] += 1; max_seen = max(max_seen, count[ch])
        while right-left+1-max_seen > k:
            count[s[left]] -= 1; left += 1
        best = max(best, right-left+1)
    return best


def contains_permutation(pattern: str, text: str) -> bool:
    """LC 567. Fixed-length window counts. O(n + m) time, O(k) space."""
    m = len(pattern)
    if m > len(text): return False
    if m == 0: return True
    need = Counter(pattern); window = Counter(text[:m])
    if window == need: return True
    for i in range(m, len(text)):
        outgoing = text[i-m]; window[outgoing] -= 1
        if not window[outgoing]: del window[outgoing]
        window[text[i]] += 1
        if window == need: return True
    return False


def total_fruit(fruits: list[int]) -> int:
    """LC 904. Longest window with <= 2 distinct types. O(n) time, O(1) space."""
    count = defaultdict(int); left = best = 0
    for right, x in enumerate(fruits):
        count[x] += 1
        while len(count) > 2:
            y = fruits[left]; count[y] -= 1; left += 1
            if count[y] == 0: del count[y]
        best = max(best, right-left+1)
    return best


def min_window(s: str, t: str) -> str:
    """LC 76. Frequency deficit and missing count. O(n+m), O(k) space."""
    if not t: return ""
    need = Counter(t); missing = len(t); left = 0; start = 0; best = float('inf')
    for right, ch in enumerate(s):
        if need[ch] > 0: missing -= 1
        need[ch] -= 1
        while missing == 0:
            if right-left+1 < best:
                best = right-left+1; start = left
            first = s[left]; need[first] += 1; left += 1
            if need[first] > 0: missing += 1
    return "" if best == float('inf') else s[start:start+best]


def max_area(height: list[int]) -> int:
    """LC 11. Move the shorter wall; the taller wall cannot improve area. O(n), O(1)."""
    i, j, best = 0, len(height)-1, 0
    while i < j:
        best = max(best, (j-i)*min(height[i], height[j]))
        if height[i] <= height[j]: i += 1
        else: j -= 1
    return best


def two_sum_sorted(nums: list[int], target: int) -> list[int]:
    """LC 167. Sorted input, 1-based indices. O(n) time, O(1) space."""
    i,j = 0,len(nums)-1
    while i < j:
        s = nums[i]+nums[j]
        if s == target: return [i+1,j+1]
        if s < target: i += 1
        else: j -= 1
    return []


def three_sum(nums: list[int]) -> list[list[int]]:
    """LC 15. Sort then complementary two pointers. O(n^2) time, O(n) sort space."""
    nums = sorted(nums); out = []
    for i, x in enumerate(nums):
        if i and x == nums[i-1]: continue
        left,right=i+1,len(nums)-1
        while left < right:
            total=x+nums[left]+nums[right]
            if total < 0: left += 1
            elif total > 0: right -= 1
            else:
                out.append([x,nums[left],nums[right]])
                left += 1; right -= 1
                while left < right and nums[left] == nums[left-1]: left += 1
    return out


def three_sum_closest(nums: list[int], target: int) -> int:
    """LC 16. O(n^2) time, O(n) sorting space."""
    nums = sorted(nums)
    if len(nums) < 3: raise ValueError('at least three values required')
    best=sum(nums[:3])
    for i in range(len(nums)-2):
        l,r=i+1,len(nums)-1
        while l < r:
            current=nums[i]+nums[l]+nums[r]
            if abs(current-target)<abs(best-target): best=current
            if current==target: return target
            if current < target: l += 1
            else: r -= 1
    return best


def trap_rain_water(height: list[int]) -> int:
    """LC 42. Two-sided waterline. O(n) time, O(1) space."""
    l,r=0,len(height)-1; left_max=right_max=0; water=0
    while l < r:
        if height[l] <= height[r]:
            left_max=max(left_max,height[l]); water+=left_max-height[l]; l+=1
        else:
            right_max=max(right_max,height[r]); water+=right_max-height[r]; r-=1
    return water


def is_palindrome(s: str) -> bool:
    """LC 125. Ignore non-alphanumeric Unicode characters. O(n), O(1)."""
    i,j=0,len(s)-1
    while i<j:
        if not s[i].isalnum(): i+=1; continue
        if not s[j].isalnum(): j-=1; continue
        if s[i].casefold()!=s[j].casefold(): return False
        i+=1; j-=1
    return True


def product_except_self(nums: list[int]) -> list[int]:
    """LC 238. Prefix and suffix products; no division. O(n), O(1) aux excluding output."""
    out=[1]*len(nums); p=1
    for i,x in enumerate(nums):out[i]=p;p*=x
    p=1
    for i in range(len(nums)-1,-1,-1):out[i]*=p;p*=nums[i]
    return out


def binary_search(nums: list[int], target: int) -> int:
    """LC 704. Invariant target, if present, is in [lo, hi). O(log n), O(1)."""
    lo,hi=0,len(nums)
    while lo<hi:
        mid=(lo+hi)//2
        if nums[mid]<target:lo=mid+1
        else:hi=mid
    return lo if lo<len(nums) and nums[lo]==target else -1


def search_rotated(nums: list[int], target: int) -> int:
    """LC 33. At least one half sorted, unique elements. O(log n), O(1)."""
    lo,hi=0,len(nums)-1
    while lo<=hi:
        mid=(lo+hi)//2
        if nums[mid]==target:return mid
        if nums[lo]<=nums[mid]:
            if nums[lo]<=target<nums[mid]:hi=mid-1
            else:lo=mid+1
        else:
            if nums[mid]<target<=nums[hi]:lo=mid+1
            else:hi=mid-1
    return -1


def search_range(nums: list[int], target: int) -> list[int]:
    """LC 34. First and last occurrence. O(log n), O(1)."""
    lo=bisect_left(nums,target);hi=bisect_right(nums,target)
    return [lo,hi-1] if lo<hi else [-1,-1]


def search_insert(nums: list[int], target: int) -> int:
    """LC 35. Lower bound. O(log n), O(1)."""
    return bisect_left(nums,target)


def find_min_rotated(nums: list[int]) -> int:
    """LC 153. Unique values, rotated ascending. O(log n), O(1)."""
    if not nums: raise ValueError('empty array')
    l,r=0,len(nums)-1
    while l<r:
        mid=(l+r)//2
        if nums[mid]>nums[r]:l=mid+1
        else:r=mid
    return nums[l]


def find_peak_element(nums: list[int]) -> int:
    """LC 162. Move toward rising slope. O(log n), O(1)."""
    if not nums: raise ValueError('empty array')
    l,r=0,len(nums)-1
    while l<r:
        m=(l+r)//2
        if nums[m]<nums[m+1]:l=m+1
        else:r=m
    return l


def min_eating_speed(piles: list[int], h: int) -> int:
    """LC 875. Binary search minimum feasible speed. O(n log M), O(1)."""
    if not piles:return 0
    if h<len(piles):raise ValueError('impossible deadline')
    lo,hi=1,max(piles)
    while lo<hi:
        mid=(lo+hi)//2
        hours=sum((p+mid-1)//mid for p in piles)
        if hours<=h:hi=mid
        else:lo=mid+1
    return lo


def ship_within_days(weights: list[int], days: int) -> int:
    """LC 1011. Monotone feasibility. O(n log(sum(weights))), O(1)."""
    if not weights:return 0
    lo,hi=max(weights),sum(weights)
    while lo<hi:
        cap=(lo+hi)//2; needed=1; load=0
        for w in weights:
            if load+w>cap:needed+=1;load=0
            load+=w
        if needed<=days:hi=cap
        else:lo=cap+1
    return lo


def count_subarray_sum(nums: list[int], k: int) -> int:
    """LC 560. Count previous prefix sums. O(n) expected time/space."""
    freq=defaultdict(int);freq[0]=1; running=answer=0
    for x in nums:
        running+=x;answer+=freq[running-k];freq[running]+=1
    return answer


def pivot_index(nums: list[int]) -> int:
    """LC 724. Left sum == total-left-current. O(n), O(1)."""
    total=sum(nums);left=0
    for i,x in enumerate(nums):
        if left==total-left-x:return i
        left+=x
    return -1


def next_greater_element(nums1: list[int], nums2: list[int]) -> list[int]:
    """LC 496. Decreasing stack mapping. O(n+m) time/space."""
    stack=[]; mapping={}
    for x in nums2:
        while stack and stack[-1]<x:mapping[stack.pop()]=x
        stack.append(x)
    return [mapping.get(x,-1) for x in nums1]


def daily_temperatures(temperatures: list[int]) -> list[int]:
    """LC 739. Stack of unresolved indices. O(n) time/space."""
    out=[0]*len(temperatures);stack=[]
    for i,x in enumerate(temperatures):
        while stack and temperatures[stack[-1]]<x:
            j=stack.pop();out[j]=i-j
        stack.append(i)
    return out


def largest_rectangle_area(heights: list[int]) -> int:
    """LC 84. Monotonic increasing stack. O(n) time/space."""
    stack=[];best=0
    for i,h in enumerate([*heights,0]):
        start=i
        while stack and stack[-1][1]>h:
            pos,height=stack.pop();best=max(best,height*(i-pos));start=pos
        stack.append((start,h))
    return best


def max_sliding_window(nums: list[int], k: int) -> list[int]:
    """LC 239. Decreasing deque of indices. O(n) time, O(k) space."""
    if k<=0:raise ValueError('k must be positive')
    d=deque();out=[]
    for i,x in enumerate(nums):
        while d and d[0]<=i-k:d.popleft()
        while d and nums[d[-1]]<=x:d.pop()
        d.append(i)
        if i>=k-1:out.append(nums[d[0]])
    return out


def longest_subarray_limit(nums: list[int], limit: int) -> int:
    """LC 1438. Two monotonic deques. O(n), O(n) space."""
    hi=deque();lo=deque();left=best=0
    for right,x in enumerate(nums):
        while hi and nums[hi[-1]]<x:hi.pop()
        while lo and nums[lo[-1]]>x:lo.pop()
        hi.append(right);lo.append(right)
        while nums[hi[0]]-nums[lo[0]]>limit:
            if hi[0]==left:hi.popleft()
            if lo[0]==left:lo.popleft()
            left+=1
        best=max(best,right-left+1)
    return best


def top_k_frequent(nums: list[int], k: int) -> list[int]:
    """LC 347. Bucket by frequency. O(n), O(n) space."""
    if k<=0:return []
    counts=Counter(nums);buckets=[[] for _ in range(len(nums)+1)]
    for number,count in counts.items():buckets[count].append(number)
    out=[]
    for b in reversed(buckets):
        out.extend(b)
        if len(out)>=k:return out[:k]
    return out


def kth_largest(nums: list[int], k: int) -> int:
    """LC 215. Min heap of k. O(n log k) time, O(k) space."""
    if not 1<=k<=len(nums):raise ValueError('invalid k')
    import heapq
    return heapq.nlargest(k,nums)[-1]


def k_closest(points: list[list[int]], k: int) -> list[list[int]]:
    """LC 973. Heap selection; order unspecified. O(n log k), O(k) heap."""
    from heapq import nsmallest
    return nsmallest(k,points,key=lambda p:p[0]*p[0]+p[1]*p[1])


def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
    """LC 56. Sort by start, merge overlap including touching endpoints. O(n log n), O(n)."""
    out=[]
    for start,end in sorted(intervals):
        if not out or start>out[-1][1]:out.append([start,end])
        else:out[-1][1]=max(out[-1][1],end)
    return out


def insert_interval(intervals: list[list[int]], new: list[int]) -> list[list[int]]:
    """LC 57. Insert into sorted nonoverlapping intervals. O(n), O(n) output."""
    out=[];i=0;n=len(intervals);start,end=new
    while i<n and intervals[i][1]<start:out.append(intervals[i][:]);i+=1
    while i<n and intervals[i][0]<=end:
        start=min(start,intervals[i][0]);end=max(end,intervals[i][1]);i+=1
    out.append([start,end]);out.extend(row[:] for row in intervals[i:]);return out


def can_attend_meetings(intervals: list[list[int]]) -> bool:
    """LC 252. Half-open meetings [start,end). O(n log n) time, O(n) sort."""
    ordered=sorted(intervals)
    return all(ordered[i-1][1]<=ordered[i][0] for i in range(1,len(ordered)))


def min_meeting_rooms(intervals: list[list[int]]) -> int:
    """LC 253. Half-open intervals; reuse room at touching times. O(n log n), O(n)."""
    import heapq
    heap=[];best=0
    for start,end in sorted(intervals):
        while heap and heap[0]<=start:heapq.heappop(heap)
        heapq.heappush(heap,end);best=max(best,len(heap))
    return best


def jump_game(nums: list[int]) -> bool:
    """LC 55. Reachable frontier. O(n), O(1)."""
    far=0
    for i,x in enumerate(nums):
        if i>far:return False
        far=max(far,i+x)
    return True


def jump_game_ii(nums: list[int]) -> int:
    """LC 45. Greedy BFS levels. O(n), O(1); -1 if unreachable."""
    if len(nums)<=1:return 0
    current=far=steps=0
    for i in range(len(nums)-1):
        if i>far:return -1
        far=max(far,i+nums[i])
        if i==current:
            steps+=1;current=far
            if current>=len(nums)-1:return steps
    return -1


def partition_labels(s: str) -> list[int]:
    """LC 763. End of each character's last occurrence. O(n), O(k)."""
    last={c:i for i,c in enumerate(s)};start=end=0;out=[]
    for i,c in enumerate(s):
        end=max(end,last[c])
        if i==end:out.append(end-start+1);start=i+1
    return out


def min_cost_climbing_stairs(cost: list[int]) -> int:
    """LC 746. Rolling DP. O(n), O(1)."""
    a=b=0
    for x in cost:a,b=b,min(a,b)+x
    return min(a,b)


def valid_parentheses(s: str) -> bool:
    """LC 20. Stack invariant: unmatched opening brackets. O(n) time/space."""
    pair={')':'(',']':'[','}':'{'};stack=[]
    for c in s:
        if c in '([{':stack.append(c)
        elif c in pair:
            if not stack or stack.pop()!=pair[c]:return False
        else:return False
    return not stack


def missing_number(nums: list[int]) -> int:
    """LC 268. XOR cancels all observed indices and values. O(n), O(1)."""
    answer=len(nums)
    for i,x in enumerate(nums):answer^=i^x
    return answer


def single_number(nums: list[int]) -> int:
    """LC 136. Every other value appears exactly twice. O(n), O(1)."""
    x=0
    for n in nums:x^=n
    return x


def hamming_weight(n: int) -> int:
    """LC 191. Number of set bits of nonnegative integer. O(log n), O(1)."""
    if n<0:raise ValueError('nonnegative input required')
    count=0
    while n:n&=n-1;count+=1
    return count


def missing_ranges(nums: list[int], lower: int, upper: int) -> list[list[int]]:
    """Generic gap detection. O(n), O(n) output, sorted distinct data required."""
    result=[];prev=lower-1
    for x in [*nums,upper+1]:
        if x-prev>=2:result.append([prev+1,x-1])
        prev=x
    return result


def range_addition(length: int, updates: list[list[int]]) -> list[int]:
    """LC 370. Difference array for inclusive range add. O(n+q), O(n) space."""
    diff=[0]*(length+1)
    for a,b,delta in updates:
        if not (0<=a<=b<length):raise ValueError('invalid update')
        diff[a]+=delta;diff[b+1]-=delta
    result=[];running=0
    for delta in diff[:length]:running+=delta;result.append(running)
    return result


def car_pooling(trips: list[list[int]], capacity: int) -> bool:
    """LC 1094. Sweep passenger changes; drop-offs occur before pickups at same stop. O(n log n)."""
    changes=defaultdict(int)
    for people,start,end in trips:
        changes[start]+=people;changes[end]-=people
    load=0
    for stop in sorted(changes):
        load+=changes[stop]
        if load>capacity:return False
    return True


def corp_flight_bookings(bookings: list[list[int]], n: int) -> list[int]:
    """LC 1109. One-indexed inclusive flight updates. O(n+q), O(n) space."""
    diff=[0]*(n+1)
    for first,last,seats in bookings:
        diff[first-1]+=seats;diff[last]-=seats
    ans=[];total=0
    for delta in diff[:n]:total+=delta;ans.append(total)
    return ans


def first_missing_positive(nums: list[int]) -> int:
    """LC 41. Position value x at index x-1. O(n) time, O(1) auxiliary; mutates input."""
    n=len(nums);i=0
    while i<n:
        x=nums[i]
        if 1<=x<=n and nums[x-1]!=x:nums[i],nums[x-1]=nums[x-1],nums[i]
        else:i+=1
    for i,x in enumerate(nums):
        if x!=i+1:return i+1
    return n+1


def find_disappeared_numbers(nums: list[int]) -> list[int]:
    """LC 448. In-place index sign marking. O(n), O(1) aux; mutates input."""
    for x in nums:nums[abs(x)-1]=-abs(nums[abs(x)-1])
    return [i+1 for i,x in enumerate(nums) if x>0]


def sorted_squares(nums: list[int]) -> list[int]:
    """LC 977. Fill output from right with larger absolute square. O(n), O(n) output."""
    i,j=0,len(nums)-1;out=[0]*len(nums)
    for pos in range(len(nums)-1,-1,-1):
        if abs(nums[i])>abs(nums[j]):out[pos]=nums[i]**2;i+=1
        else:out[pos]=nums[j]**2;j-=1
    return out


def remove_duplicates_sorted(nums: list[int]) -> int:
    """LC 26. Mutates prefix in-place; return its length. O(n), O(1)."""
    write=0
    for x in nums:
        if write==0 or x!=nums[write-1]:nums[write]=x;write+=1
    return write


def sort_colors(nums: list[int]) -> None:
    """LC 75. Dutch national flag, values 0,1,2. O(n), O(1), mutates input."""
    low=i=0;high=len(nums)-1
    while i<=high:
        if nums[i]==0:nums[low],nums[i]=nums[i],nums[low];low+=1;i+=1
        elif nums[i]==1:i+=1
        elif nums[i]==2:nums[i],nums[high]=nums[high],nums[i];high-=1
        else:raise ValueError('expected only 0, 1, 2')


def find_anagrams(s: str,p: str) -> list[int]:
    """LC 438. Frequency-matched windows. O(n+m), O(k) space."""
    m=len(p)
    if m==0 or m>len(s):return []
    need=Counter(p);window=Counter(s[:m]);out=[]
    if window==need:out.append(0)
    for end in range(m,len(s)):
        c=s[end-m];window[c]-=1
        if window[c]==0:del window[c]
        window[s[end]]+=1
        if window==need:out.append(end-m+1)
    return out


def max_average_subarray(nums: list[int], k: int) -> float:
    """LC 643. Fixed-size rolling sum. O(n), O(1)."""
    if not 1<=k<=len(nums):raise ValueError('invalid k')
    total=sum(nums[:k]);best=total
    for i in range(k,len(nums)):
        total+=nums[i]-nums[i-k];best=max(best,total)
    return best/k


def reverse_bits(n: int) -> int:
    """LC 190. Fixed unsigned 32-bit reversal. O(1), O(1)."""
    n &= (1<<32)-1;ans=0
    for _ in range(32):ans=(ans<<1)|(n&1);n>>=1
    return ans


def maximal_rectangle(matrix:list[list[str]])->int:
    """LC 85. Histogram per row. O(RC) time, O(C) working space."""
    if not matrix:return 0
    heights=[0]*len(matrix[0]);best=0
    for row in matrix:
        for j,x in enumerate(row):heights[j]=heights[j]+1 if str(x)=='1' else 0
        best=max(best,largest_rectangle_area(heights))
    return best


def erase_overlap_intervals(intervals:list[list[int]])->int:
    """LC 435. Keep interval with smallest end. O(n log n) time, O(n) sort space."""
    if not intervals:return 0
    chosen_end=float('-inf');removed=0
    for start,end in sorted(intervals,key=lambda x:x[1]):
        if start>=chosen_end:chosen_end=end
        else:removed+=1
    return removed


def find_min_arrow_shots(points:list[list[int]])->int:
    """LC 452. Closed balloon intervals; shoot at earliest ending point. O(n log n), O(n) sort."""
    arrows=0;end=float('-inf')
    for start,stop in sorted(points,key=lambda x:x[1]):
        if arrows==0 or start>end:arrows+=1;end=stop
    return arrows


def next_greater_elements_circular(nums:list[int])->list[int]:
    """LC 503. Twice scan indices in cyclic array, push first-pass only. O(n), O(n)."""
    n=len(nums);ans=[-1]*n;stack=[]
    for i in range(2*n):
        while stack and nums[stack[-1]]<nums[i%n]:ans[stack.pop()]=nums[i%n]
        if i<n:stack.append(i)
    return ans


def top_k_frequent_words(words:list[str],k:int)->list[str]:
    """LC 692. Sort by -count, then alphabetical. O(n + u log u), O(u)."""
    counts=Counter(words)
    return sorted(counts,key=lambda x:(-counts[x],x))[:k]


def last_stone_weight(stones:list[int])->int:
    """LC 1046. Max-heap by negation. O(n log n), O(n)."""
    import heapq
    heap=[-x for x in stones];heapq.heapify(heap)
    while len(heap)>1:
        a=-heapq.heappop(heap);b=-heapq.heappop(heap)
        if a>b:heapq.heappush(heap,-(a-b))
    return -heap[0] if heap else 0
