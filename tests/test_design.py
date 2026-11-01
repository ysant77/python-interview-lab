import pytest
from interview_patterns import design as d


def test_numarray():
    n=d.NumArray([-2,0,3,-5,2,-1]);assert n.sumRange(0,2)==1;assert n.sumRange(2,5)==-1

def test_nummatrix():
    n=d.NumMatrix([[3,0,1,4,2],[5,6,3,2,1],[1,2,0,1,5],[4,1,0,1,7],[1,0,3,0,5]])
    assert n.sumRegion(2,1,4,3)==8
    assert n.sumRegion(1,1,2,2)==11

def test_hashset_hashmap():
    st=d.MyHashSet();st.add(1);st.add(1032);st.add(-1)
    assert st.contains(1) and st.contains(1032) and not st.contains(2)
    st.remove(1);assert not st.contains(1)
    hm=d.MyHashMap();hm.put(1,7);hm.put(1032,9);hm.put(1,3)
    assert hm.get(1)==3 and hm.get(1032)==9 and hm.get(2)==-1
    hm.remove(1032);assert hm.get(1032)==-1

def test_codec():
    codec=d.Codec()
    for words in [[],[''],['hello','world'],['#','##','a#0', '', '🌏']]:
        assert codec.decode(codec.encode(words))==words
    with pytest.raises(ValueError):codec.decode('2#x')

def test_randomizedset():
    s=d.RandomizedSet()
    assert s.insert(1) and not s.insert(1)
    assert s.insert(2) and s.remove(1) and not s.remove(1)
    assert s.getRandom()==2

def test_stack_queue():
    s=d.MyStack();s.push(1);s.push(2)
    assert s.top()==2 and s.pop()==2 and s.pop()==1 and s.empty()
    q=d.MyQueue();q.push(1);q.push(2)
    assert q.peek()==1 and q.pop()==1 and q.pop()==2 and q.empty()

def test_min_stack_and_spanner():
    st=d.MinStack();st.push(-2);st.push(0);st.push(-3)
    assert st.getMin()==-3
    st.pop();assert st.top()==0 and st.getMin()==-2
    sp=d.StockSpanner()
    assert [sp.next(x) for x in [100,80,60,70,60,75,85]]==[1,1,1,2,1,4,6]

def test_freqstack():
    st=d.FreqStack()
    for x in [5,7,5,7,4,5]:st.push(x)
    assert [st.pop() for _ in range(4)]==[5,7,5,4]

def test_timemap():
    tm=d.TimeMap();tm.set('foo','bar',1)
    assert tm.get('foo',1)=='bar' and tm.get('foo',3)=='bar'
    tm.set('foo','bar2',4)
    assert tm.get('foo',4)=='bar2' and tm.get('foo',5)=='bar2' and tm.get('x',99)==''
    with pytest.raises(ValueError):tm.set('foo','older',3)

def test_linkedlist():
    ll=d.MyLinkedList();ll.addAtHead(1);ll.addAtTail(3);ll.addAtIndex(1,2)
    assert ll.get(1)==2
    ll.deleteAtIndex(1)
    assert ll.get(1)==3
    ll.deleteAtIndex(9);assert ll.size==2

def test_browser_history():
    b=d.BrowserHistory('leetcode.com')
    for x in ['google.com','facebook.com','youtube.com']:b.visit(x)
    assert b.back(1)=='facebook.com' and b.back(1)=='google.com'
    assert b.forward(1)=='facebook.com'
    b.visit('linkedin.com')
    assert b.forward(2)=='linkedin.com' and b.back(2)=='google.com'

def test_circular_queue():
    q=d.MyCircularQueue(3)
    assert q.enQueue(1) and q.enQueue(2) and q.enQueue(3)
    assert not q.enQueue(4) and q.Rear()==3 and q.isFull()
    assert q.deQueue() and q.enQueue(4) and q.Rear()==4 and q.Front()==2

def test_lru():
    lru=d.LRUCache(2);lru.put(1,1);lru.put(2,2)
    assert lru.get(1)==1
    lru.put(3,3);assert lru.get(2)==-1
    lru.put(4,4);assert lru.get(1)==-1 and lru.get(3)==3 and lru.get(4)==4

def test_median_and_kth_largest():
    med=d.MedianFinder();med.addNum(1);med.addNum(2)
    assert med.findMedian()==1.5
    med.addNum(3);assert med.findMedian()==2.0
    kth=d.KthLargest(3,[4,5,8,2])
    assert [kth.add(x) for x in [3,5,10,9,4]]==[4,5,5,8,8]
