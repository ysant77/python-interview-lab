"""15 interview API-design exercises, plus LRU cache and streaming median.

Names mirror platform class interfaces where practical. Complexity statements
refer to typical operations; Python dictionaries have expected O(1) operations.
"""
from __future__ import annotations
from bisect import bisect_right
from collections import defaultdict,deque,OrderedDict
from heapq import heappop,heappush
from random import choice

class NumArray:
    """LC 303. Init O(n) time/space; sumRange O(1) time."""
    def __init__(self, nums:list[int]):
        self.prefix=[0]
        for x in nums:self.prefix.append(self.prefix[-1]+x)
    def sumRange(self,left:int,right:int)->int:
        return self.prefix[right+1]-self.prefix[left]

class NumMatrix:
    """LC 304. Init O(RC) time/space; sumRegion O(1) time."""
    def __init__(self,matrix:list[list[int]]):
        rows=len(matrix);cols=len(matrix[0]) if matrix else 0
        self.prefix=[[0]*(cols+1) for _ in range(rows+1)]
        for r in range(rows):
            for c in range(cols):
                self.prefix[r+1][c+1]=matrix[r][c]+self.prefix[r][c+1]+self.prefix[r+1][c]-self.prefix[r][c]
    def sumRegion(self,row1:int,col1:int,row2:int,col2:int)->int:
        p=self.prefix
        return p[row2+1][col2+1]-p[row1][col2+1]-p[row2+1][col1]+p[row1][col1]

class MyHashSet:
    """LC 705. Chained hash buckets; expected O(1) lookup/add/remove, O(n) space."""
    def __init__(self):self.buckets=[set() for _ in range(1031)]
    def add(self,key:int)->None:self.buckets[key%len(self.buckets)].add(key)
    def remove(self,key:int)->None:self.buckets[key%len(self.buckets)].discard(key)
    def contains(self,key:int)->bool:return key in self.buckets[key%len(self.buckets)]

class MyHashMap:
    """LC 706. Chained buckets with a fixed bucket count; expected O(1), worst O(n)."""
    def __init__(self):self.buckets=[[] for _ in range(1031)]
    def put(self,key:int,value:int)->None:
        bucket=self.buckets[key%len(self.buckets)]
        for i,(k,_) in enumerate(bucket):
            if k==key:bucket[i]=(key,value);return
        bucket.append((key,value))
    def get(self,key:int)->int:
        for k,v in self.buckets[key%len(self.buckets)]:
            if k==key:return v
        return -1
    def remove(self,key:int)->None:
        bucket=self.buckets[key%len(self.buckets)]
        for i,(k,_) in enumerate(bucket):
            if k==key:bucket.pop(i);return

class Codec:
    """LC 271. Length-prefixed lossless string serialization; O(total chars)."""
    def encode(self,strs:list[str])->str:return ''.join(f'{len(s)}#{s}' for s in strs)
    def decode(self,s:str)->list[str]:
        out=[];i=0
        while i<len(s):
            sep=s.find('#',i)
            if sep<0 or not s[i:sep].isdigit():raise ValueError('invalid length prefix')
            n=int(s[i:sep]);start=sep+1;end=start+n
            if end>len(s):raise ValueError('truncated payload')
            out.append(s[start:end]);i=end
        return out

class RandomizedSet:
    """LC 380. Index map + compact vector. Expected O(1) insert/remove/random."""
    def __init__(self):self.data=[];self.index={}
    def insert(self,val:int)->bool:
        if val in self.index:return False
        self.index[val]=len(self.data);self.data.append(val);return True
    def remove(self,val:int)->bool:
        if val not in self.index:return False
        idx=self.index.pop(val);last=self.data[-1]
        self.data[idx]=last;self.data.pop()
        if idx<len(self.data):self.index[last]=idx
        return True
    def getRandom(self)->int:return choice(self.data)

class MyStack:
    """LC 225. One queue; push O(n), pop/top O(1), O(n) space."""
    def __init__(self):self.q=deque()
    def push(self,x:int)->None:
        self.q.append(x)
        for _ in range(len(self.q)-1):self.q.append(self.q.popleft())
    def pop(self)->int:return self.q.popleft()
    def top(self)->int:return self.q[0]
    def empty(self)->bool:return not self.q

class MyQueue:
    """LC 232. Two stacks; push O(1), pop/peek amortized O(1), worst O(n)."""
    def __init__(self):self.in_stack=[];self.out_stack=[]
    def push(self,x:int)->None:self.in_stack.append(x)
    def _transfer(self)->None:
        if not self.out_stack:
            while self.in_stack:self.out_stack.append(self.in_stack.pop())
    def pop(self)->int:self._transfer();return self.out_stack.pop()
    def peek(self)->int:self._transfer();return self.out_stack[-1]
    def empty(self)->bool:return not(self.in_stack or self.out_stack)

class MinStack:
    """LC 155. Track duplicate minima. Push/pop/top/getMin O(1), O(n) space."""
    def __init__(self):self.data=[]
    def push(self,val:int)->None:self.data.append((val,min(val,self.data[-1][1]) if self.data else val))
    def pop(self)->None:self.data.pop()
    def top(self)->int:return self.data[-1][0]
    def getMin(self)->int:return self.data[-1][1]

class StockSpanner:
    """LC 901. Monotone stack; next O(1) amortized, worst O(n), O(n) storage."""
    def __init__(self):self.stack=[]
    def next(self,price:int)->int:
        span=1
        while self.stack and self.stack[-1][0]<=price:span+=self.stack.pop()[1]
        self.stack.append((price,span));return span

class FreqStack:
    """LC 895. Frequency-bucket stack. Push/pop O(1) expected, O(n) space."""
    def __init__(self):self.freq=defaultdict(int);self.groups=defaultdict(list);self.max_freq=0
    def push(self,val:int)->None:
        self.freq[val]+=1;f=self.freq[val];self.max_freq=max(self.max_freq,f);self.groups[f].append(val)
    def pop(self)->int:
        val=self.groups[self.max_freq].pop();self.freq[val]-=1
        if not self.groups[self.max_freq]:self.max_freq-=1
        return val

class TimeMap:
    """LC 981. For nondecreasing timestamps per key: set amortized O(1), get O(log n)."""
    def __init__(self):self.data=defaultdict(list)
    def set(self,key:str,value:str,timestamp:int)->None:
        history=self.data[key]
        if history and timestamp<history[-1][0]:raise ValueError('timestamps must be nondecreasing')
        history.append((timestamp,value))
    def get(self,key:str,timestamp:int)->str:
        history=self.data[key];i=bisect_right(history,(timestamp,chr(0x10FFFF)))-1
        return history[i][1] if i>=0 else ''

class _DLLNode:
    __slots__=('val','next')
    def __init__(self,val:int=0,next=None):self.val=val;self.next=next

class MyLinkedList:
    """LC 707. Singly linked sentinel. get/add/delete O(index), O(n) storage."""
    def __init__(self):self.head=_DLLNode();self.size=0
    def _before(self,index:int):
        cur=self.head
        for _ in range(index):cur=cur.next
        return cur
    def get(self,index:int)->int:
        if not 0<=index<self.size:return -1
        return self._before(index).next.val
    def addAtHead(self,val:int)->None:self.addAtIndex(0,val)
    def addAtTail(self,val:int)->None:self.addAtIndex(self.size,val)
    def addAtIndex(self,index:int,val:int)->None:
        if index>self.size:return
        index=max(index,0);prev=self._before(index);prev.next=_DLLNode(val,prev.next);self.size+=1
    def deleteAtIndex(self,index:int)->None:
        if not 0<=index<self.size:return
        prev=self._before(index);prev.next=prev.next.next;self.size-=1

class BrowserHistory:
    """LC 1472. Visit amortized O(1), back/forward O(1), O(n) history."""
    def __init__(self,homepage:str):self.history=[homepage];self.pos=0
    def visit(self,url:str)->None:
        del self.history[self.pos+1:];self.history.append(url);self.pos+=1
    def back(self,steps:int)->str:self.pos=max(0,self.pos-steps);return self.history[self.pos]
    def forward(self,steps:int)->str:self.pos=min(len(self.history)-1,self.pos+steps);return self.history[self.pos]

class MyCircularQueue:
    """LC 622. Ring buffer; enQueue/deQueue/Front/Rear O(1), O(k) space."""
    def __init__(self,k:int):
        if k<=0:raise ValueError('capacity must be positive')
        self.data=[None]*k;self.head=0;self.size=0
    def enQueue(self,value:int)->bool:
        if self.isFull():return False
        self.data[(self.head+self.size)%len(self.data)]=value;self.size+=1;return True
    def deQueue(self)->bool:
        if self.isEmpty():return False
        self.data[self.head]=None;self.head=(self.head+1)%len(self.data);self.size-=1;return True
    def Front(self)->int:return -1 if self.isEmpty() else self.data[self.head]
    def Rear(self)->int:return -1 if self.isEmpty() else self.data[(self.head+self.size-1)%len(self.data)]
    def isEmpty(self)->bool:return self.size==0
    def isFull(self)->bool:return self.size==len(self.data)

class LRUCache:
    """LC 146. Ordered mapping; get/put expected O(1), O(capacity) storage."""
    def __init__(self,capacity:int):
        if capacity<=0:raise ValueError('positive capacity required')
        self.capacity=capacity;self.data=OrderedDict()
    def get(self,key:int)->int:
        if key not in self.data:return -1
        self.data.move_to_end(key);return self.data[key]
    def put(self,key:int,value:int)->None:
        if key in self.data:self.data.move_to_end(key)
        self.data[key]=value
        if len(self.data)>self.capacity:self.data.popitem(last=False)

class MedianFinder:
    """LC 295. Two heaps; add O(log n), median O(1), O(n) space."""
    def __init__(self):self.left=[];self.right=[]
    def addNum(self,num:int)->None:
        heappush(self.left,-num)
        heappush(self.right,-heappop(self.left))
        if len(self.right)>len(self.left):heappush(self.left,-heappop(self.right))
    def findMedian(self)->float:
        if not self.left:raise ValueError('no numbers')
        if len(self.left)>len(self.right):return float(-self.left[0])
        return (-self.left[0]+self.right[0])/2

class KthLargest:
    """LC 703. Min-heap of k values. add O(log k), O(k) storage."""
    def __init__(self,k:int,nums:list[int]):
        if k<=0:raise ValueError('k must be positive')
        self.k=k;self.heap=[]
        for val in nums:self.add(val)
    def add(self,val:int)->int:
        heappush(self.heap,val)
        if len(self.heap)>self.k:heappop(self.heap)
        return self.heap[0]
