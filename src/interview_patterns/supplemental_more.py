"""Extra DSA/DP source-linked problems using independent Python solutions."""
from __future__ import annotations
from bisect import bisect_left
from collections import Counter, defaultdict, deque
from functools import lru_cache
from heapq import heappop, heappush
from itertools import combinations
from math import inf


def max_length_ribbons(ribbons:list[int],k:int)->int:
    """LC 1891 | Binary search feasible equal piece length. T O(n log max_ribbon), S O(1)."""
    if not ribbons or k<1:return 0
    lo,hi=1,max(ribbons);best=0
    while lo<=hi:
        mid=(lo+hi)//2
        if sum(length//mid for length in ribbons)>=k:best=mid;lo=mid+1
        else:hi=mid-1
    return best


def fibonacci(n:int)->int:
    """LC 509 | Iterative Fibonacci. T O(n), S O(1)."""
    if n<0:raise ValueError('n must be nonnegative')
    a,b=0,1
    for _ in range(n):a,b=b,a+b
    return a


def find_closest_elements(arr:list[int],k:int,x:int)->list[int]:
    """LC 658 | Binary search k-window left edge. T O(log(n-k)+k), S O(k)."""
    if not 0<=k<=len(arr):raise ValueError('invalid k')
    lo,hi=0,len(arr)-k
    while lo<hi:
        mid=(lo+hi)//2
        if x-arr[mid]>arr[mid+k]-x:lo=mid+1
        else:hi=mid
    return arr[lo:lo+k]


def is_interleave(s1:str,s2:str,s3:str)->bool:
    """LC 97 | Dynamic programming matching both input streams. T O(n*m), S O(m)."""
    if len(s1)+len(s2)!=len(s3):return False
    dp=[False]*(len(s2)+1);dp[0]=True
    for j in range(1,len(dp)):dp[j]=dp[j-1] and s2[j-1]==s3[j-1]
    for i in range(1,len(s1)+1):
        dp[0]=dp[0] and s1[i-1]==s3[i-1]
        for j in range(1,len(dp)):
            dp[j]=(dp[j] and s1[i-1]==s3[i+j-1]) or (dp[j-1] and s2[j-1]==s3[i+j-1])
    return dp[-1]


def largest_subarray(nums:list[int],k:int)->list[int]:
    """LC 1708 | Lexicographically largest length-k subarray. T O(n*k), S O(k)."""
    if not 1<=k<=len(nums):raise ValueError('invalid k')
    return max((nums[i:i+k] for i in range(len(nums)-k+1)))


def find_least_num_unique_ints(arr:list[int],k:int)->int:
    """LC 1481 | Remove smallest frequencies first. T O(n log n), S O(n)."""
    counts=sorted(Counter(arr).values());remaining=len(counts)
    for freq in counts:
        if k<freq:break
        k-=freq;remaining-=1
    return remaining


def longest_palindrome_subseq(s:str)->int:
    """LC 516 | Interval DP: matching ends add 2. T O(n²), S O(n²)."""
    n=len(s)
    if not n:return 0
    dp=[[0]*n for _ in range(n)]
    for i in range(n-1,-1,-1):
        dp[i][i]=1
        for j in range(i+1,n):
            dp[i][j]=2+(dp[i+1][j-1] if j>i+1 else 0) if s[i]==s[j] else max(dp[i+1][j],dp[i][j-1])
    return dp[0][n-1]


def longest_palindrome(s:str)->str:
    """LC 5 | Expand around each center. T O(n²), S O(1)."""
    best=(0,0)
    for i in range(len(s)):
        for l,r in ((i,i),(i,i+1)):
            while l>=0 and r<len(s) and s[l]==s[r]:l-=1;r+=1
            if r-l-2>best[1]-best[0]:best=(l+1,r-1)
    return s[best[0]:best[1]+1]


def length_of_longest_substring_k_distinct(s:str,k:int)->int:
    """LC 340 | Sliding window with char counter. T O(n), S O(min(n,k))."""
    if k<=0:return 0
    freq=Counter();left=0;best=0
    for right,ch in enumerate(s):
        freq[ch]+=1
        while len(freq)>k:
            out=s[left];freq[out]-=1
            if freq[out]==0:del freq[out]
            left+=1
        best=max(best,right-left+1)
    return best


def length_of_longest_substring_two_distinct(s:str)->int:
    """LC 159 | Two distinct characters, specialized k-distinct window. T O(n), S O(1)."""
    return length_of_longest_substring_k_distinct(s,2)


def longest_ones(nums:list[int],k:int)->int:
    """LC 1004 | Longest binary window containing <= k zeros. T O(n), S O(1)."""
    left=zeros=best=0
    for right,value in enumerate(nums):
        zeros+=int(value==0)
        while zeros>k:
            zeros-=int(nums[left]==0);left+=1
        best=max(best,right-left+1)
    return best


def max_taxi_earnings(n:int,rides:list[list[int]])->int:
    """LC 2008 | Weighted intervals ending at each coordinate. T O(n+R), S O(n+R)."""
    ends=defaultdict(list)
    for start,end,tip in rides:ends[end].append((start,end-start+tip))
    dp=[0]*(n+1)
    for end in range(1,n+1):
        dp[end]=max([dp[end-1]]+[dp[start]+profit for start,profit in ends[end]])
    return dp[-1]


def find_median_sorted_arrays(nums1:list[int],nums2:list[int])->float:
    """LC 4 | Binary partition smaller array. T O(log(min(n,m)+1)), S O(1)."""
    if len(nums1)>len(nums2):return find_median_sorted_arrays(nums2,nums1)
    n,m=len(nums1),len(nums2)
    if n+m==0:raise ValueError('at least one value required')
    left,right=0,n;left_count=(n+m+1)//2
    while left<=right:
        i=(left+right)//2;j=left_count-i
        a_l=nums1[i-1] if i else -inf;a_r=nums1[i] if i<n else inf
        b_l=nums2[j-1] if j else -inf;b_r=nums2[j] if j<m else inf
        if a_l<=b_r and b_l<=a_r:
            if (n+m)%2:return float(max(a_l,b_l))
            return (max(a_l,b_l)+min(a_r,b_r))/2
        if a_l>b_r:right=i-1
        else:left=i+1
    raise ValueError('arrays must be sorted')


def min_insertions_palindrome(s:str)->int:
    """LC 1312 | len(s) - longest palindromic subsequence. T O(n²), S O(n²)."""
    return len(s)-longest_palindrome_subseq(s)


def min_cut_palindrome(s:str)->int:
    """LC 132 | Precompute palindrome intervals, DP min cuts. T/S O(n²)."""
    n=len(s)
    if n==0:return 0
    palindrome=[[False]*n for _ in range(n)]
    for i in range(n-1,-1,-1):
        for j in range(i,n):
            palindrome[i][j]=s[i]==s[j] and (j-i<2 or palindrome[i+1][j-1])
    cuts=[0]*n
    for j in range(n):
        cuts[j]=0 if palindrome[0][j] else min(cuts[i-1]+1 for i in range(1,j+1) if palindrome[i][j])
    return cuts[-1]


def count_palindromic_substrings(s:str)->int:
    """LC 647 | Expand around centers. T O(n²), S O(1)."""
    total=0
    for center in range(len(s)):
        for left,right in ((center,center),(center,center+1)):
            while left>=0 and right<len(s) and s[left]==s[right]:
                total+=1;left-=1;right+=1
    return total


def minimum_difference(nums:list[int])->int:
    """LC 2035 | Meet-in-middle equal-cardinality subset sums. T O(n*2^n), S O(2^n)."""
    count=len(nums)//2
    if len(nums)%2:raise ValueError('requires 2n elements')
    left,right=nums[:count],nums[count:]
    sums_l=defaultdict(list);sums_r=defaultdict(list)
    for take in range(count+1):
        sums_l[take]=[sum(v) for v in combinations(left,take)]
        sums_r[take]=sorted(sum(v) for v in combinations(right,take))
    total=sum(nums);answer=inf
    for take,values in sums_l.items():
        choices=sums_r[count-take]
        for current in values:
            target=total/2-current
            pos=bisect_left(choices,target)
            for j in (pos-1,pos):
                if 0<=j<len(choices):answer=min(answer,abs(total-2*(current+choices[j])))
    return int(answer)


def rearrange_string_k_distance_apart(s:str,k:int)->str:
    """LC 358 | Max-frequency heap with k-step cooldown queue. T O(n log alphabet), S O(alphabet+k)."""
    if k<=1:return s
    heap=[(-count,ch) for ch,count in Counter(s).items()]
    from heapq import heapify
    heapify(heap);wait=deque();result=[]
    for t in range(len(s)):
        while wait and wait[0][0]<=t:
            _,count,ch=wait.popleft();heappush(heap,(count,ch))
        if not heap:return ''
        neg_count,ch=heappop(heap);result.append(ch)
        if neg_count+1<0:wait.append((t+k,neg_count+1,ch))
    return ''.join(result)


def reorganize_string(s:str)->str:
    """LC 767 | Greedy max-frequency nonadjacent arrangement. T O(n log alphabet), S O(alphabet)."""
    from heapq import heapify
    heap=[(-frequency,char) for char,frequency in Counter(s).items()]
    heapify(heap);result=[];previous=(0,'')
    while heap:
        freq,ch=heappop(heap);result.append(ch)
        if previous[0]<0:heappush(heap,previous)
        previous=(freq+1,ch)
    return ''.join(result) if len(result)==len(s) else ''


def shortest_common_supersequence(str1:str,str2:str)->str:
    """LC 1092 | LCS DP and reverse reconstruction. T/S O(n*m)."""
    n,m=len(str1),len(str2);dp=[[0]*(m+1) for _ in range(n+1)]
    for i in range(n-1,-1,-1):
        for j in range(m-1,-1,-1):
            dp[i][j]=1+dp[i+1][j+1] if str1[i]==str2[j] else max(dp[i+1][j],dp[i][j+1])
    i=j=0;out=[]
    while i<n and j<m:
        if str1[i]==str2[j]:out.append(str1[i]);i+=1;j+=1
        elif dp[i+1][j]>=dp[i][j+1]:out.append(str1[i]);i+=1
        else:out.append(str2[j]);j+=1
    return ''.join(out)+str1[i:]+str2[j:]


def frequency_sort(s:str)->str:
    """LC 451 | Sort by descending character count. T O(n + A log A), S O(A+n)."""
    return ''.join(char*freq for char,freq in sorted(Counter(s).items(),key=lambda x:(-x[1],x[0])))


def find_substring_concat(s:str,words:list[str])->list[int]:
    """LC 30 | Offset-based word-size sliding window. T O(n*word_len), S O(unique words)."""
    if not s or not words or not words[0]:return []
    width=len(words[0]);total=len(words)
    if any(len(word)!=width for word in words):raise ValueError('uniform word length required')
    need=Counter(words);result=[]
    for offset in range(width):
        left=offset;seen=Counter();used=0
        for right in range(offset,len(s)-width+1,width):
            word=s[right:right+width]
            if word not in need:
                seen.clear();used=0;left=right+width;continue
            seen[word]+=1;used+=1
            while seen[word]>need[word]:
                old=s[left:left+width];seen[old]-=1;used-=1;left+=width
            if used==total:
                result.append(left)
                old=s[left:left+width];seen[old]-=1;used-=1;left+=width
    return sorted(result)


def is_valid_palindrome_k(s:str,k:int)->bool:
    """LC 1216 | At most k deletions iff n-LPS<=k. T/S O(n²)."""
    return len(s)-longest_palindrome_subseq(s)<=k
