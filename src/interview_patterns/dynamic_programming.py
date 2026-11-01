"""DP and other pattern examples. State transition before implementation."""
from __future__ import annotations
from bisect import bisect_left
from functools import lru_cache


def climb_stairs(n: int) -> int:
    """LC 70. ways(n) = ways(n-1)+ways(n-2). O(n) time, O(1) space."""
    if n<0:raise ValueError('n must be nonnegative')
    a,b=1,1
    for _ in range(n):a,b=b,a+b
    return a


def rob(nums: list[int]) -> int:
    """LC 198. Either skip or rob this house. O(n), O(1)."""
    skip=take=0
    for x in nums:skip,take=max(skip,take),skip+x
    return max(skip,take)


def rob_circular(nums: list[int]) -> int:
    """LC 213. First and last cannot both be selected. O(n), O(1)."""
    if len(nums)<=1:return nums[0] if nums else 0
    return max(rob(nums[:-1]),rob(nums[1:]))


def coin_change(coins: list[int],amount: int) -> int:
    """LC 322. Minimum unbounded coins. O(amount*len(coins)), O(amount)."""
    if any(c<=0 for c in coins):raise ValueError('coins must be positive')
    dp=[amount+1]*(amount+1);dp[0]=0
    for s in range(1,amount+1):
        for c in coins:
            if c<=s:dp[s]=min(dp[s],dp[s-c]+1)
    return -1 if dp[amount]>amount else dp[amount]


def coin_change_ii(amount: int,coins: list[int]) -> int:
    """LC 518. Number of unordered coin combinations. O(amount*C), O(amount)."""
    if any(c<=0 for c in coins):raise ValueError('coins must be positive')
    dp=[0]*(amount+1);dp[0]=1
    for c in coins:
        for s in range(c,amount+1):dp[s]+=dp[s-c]
    return dp[amount]


def num_squares(n: int) -> int:
    """LC 279. Fewest perfect squares. O(n*sqrt(n)) time, O(n) space."""
    dp=[0]+[n+1]*n
    for i in range(1,n+1):
        j=1
        while j*j<=i:dp[i]=min(dp[i],dp[i-j*j]+1);j+=1
    return dp[n]


def length_of_lis(nums: list[int]) -> int:
    """LC 300. Patience sorting tails. O(n log n), O(n) space."""
    tails=[]
    for x in nums:
        i=bisect_left(tails,x)
        if i==len(tails):tails.append(x)
        else:tails[i]=x
    return len(tails)


def can_partition(nums: list[int]) -> bool:
    """LC 416. 0/1 subset sum on nonnegative values. O(n*sum), O(sum)."""
    if any(x<0 for x in nums):raise ValueError('nonnegative values required')
    total=sum(nums)
    if total%2:return False
    target=total//2;reachable={0}
    for x in nums:
        reachable|={s+x for s in reachable if s+x<=target}
    return target in reachable


def target_sum(nums: list[int],target: int) -> int:
    """LC 494. Signed sum counts with Counter DP. O(n*range), O(range)."""
    from collections import Counter
    counts=Counter({0:1})
    for x in nums:
        updated=Counter()
        for s,cnt in counts.items():
            updated[s+x]+=cnt;updated[s-x]+=cnt
        counts=updated
    return counts[target]


def edit_distance(word1: str,word2: str) -> int:
    """LC 72. Insert/delete/substitute. O(mn) time, O(n) space."""
    prev=list(range(len(word2)+1))
    for i,a in enumerate(word1,1):
        curr=[i]
        for j,b in enumerate(word2,1):
            curr.append(prev[j-1] if a==b else min(prev[j],curr[-1],prev[j-1])+1)
        prev=curr
    return prev[-1]


def knapsack_01(weights: list[int],values: list[int],capacity: int) -> int:
    """Classic 0/1 knapsack. O(n*capacity), O(capacity)."""
    if len(weights)!=len(values) or any(w<0 for w in weights):raise ValueError('invalid weights/values')
    dp=[0]*(capacity+1)
    for weight,value in zip(weights,values):
        for cap in range(capacity,weight-1,-1):dp[cap]=max(dp[cap],dp[cap-weight]+value)
    return dp[capacity]


def unbounded_knapsack(weights: list[int],values: list[int],capacity: int) -> int:
    """Unbounded variant; positive weights required. O(n*capacity), O(capacity)."""
    if len(weights)!=len(values) or any(w<=0 for w in weights):raise ValueError('invalid weights/values')
    dp=[0]*(capacity+1)
    for weight,value in zip(weights,values):
        for cap in range(weight,capacity+1):dp[cap]=max(dp[cap],dp[cap-weight]+value)
    return dp[capacity]


def longest_common_subsequence(a: str,b: str) -> int:
    """LC 1143. LCS dynamic programming. O(mn), O(n)."""
    previous=[0]*(len(b)+1)
    for x in a:
        current=[0]
        for j,y in enumerate(b,1):current.append(previous[j-1]+1 if x==y else max(current[-1],previous[j]))
        previous=current
    return previous[-1]


def unique_paths(m:int,n:int)->int:
    """LC 62. Number of grid paths with right/down steps. O(mn), O(n)."""
    if m<=0 or n<=0:return 0
    dp=[1]*n
    for _ in range(m-1):
        for col in range(1,n):dp[col]+=dp[col-1]
    return dp[-1]
