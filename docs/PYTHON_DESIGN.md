# Python Data Structure / API Design Drills

The design questions test *invariants* and client-visible operation guarantees, not merely implementing methods with their expected names. The primary reference module is [`design.py`](../src/interview_patterns/design.py).

| Problem / LC | Structure and invariant | Typical time | Space |
|---|---|---|---|
| 303 Range Sum | 1D prefix sum, `prefix[0]=0` | build O(n), query O(1) | O(n) |
| 304 Range Sum 2D | padded integral image; inclusion–exclusion | build O(RC), query O(1) | O(RC) |
| 705 HashSet | bucketed sets (fixed 1031 buckets) | expected O(1), pathological O(n) | O(n) |
| 706 HashMap | buckets storing key-value pairs | expected O(1) at bounded load; no resizing here | O(n) |
| 271 Encode/Decode | length prefix + delimiter to disambiguate | O(total characters) | O(total characters) |
| 380 RandomizedSet | dense array + mapping from value to index | expected O(1) add/remove/random | O(n) |
| 225 Stack via Queues | rotate on push to keep top at queue head | push O(n); pop O(1) | O(n) |
| 232 Queue via Stacks | amortized transfer between input/output stacks | amortized O(1), worst O(n) | O(n) |
| 155 MinStack | store current minimum alongside each pushed value | O(1) operations | O(n) |
| 901 StockSpanner | decreasing monotonic stack with spans | amortized O(1) per call | O(n) |
| 895 FreqStack | value frequency + per-frequency recency stacks | O(1) add/remove | O(n) |
| 981 TimeMap | append time-ordered versions; binary-search lookup | set amortized O(1), get O(log n) | O(n) |
| 707 LinkedList | sentinel node, size invariant | get/insert/delete O(n) | O(n) |
| 1472 BrowserHistory | array + cursor; truncate forward on new visit | back/forward O(1), visit amortized O(1) | O(n) |
| 622 CircularQueue | fixed capacity + head index + count | O(1) operations | O(k) |

**Important limitations:** This library's toy `MyHashMap` does not dynamically resize or offer production worst-case hashing guarantees. `TimeMap` assumes nondecreasing timestamps for each key; otherwise binary search on appended values is invalid. For interviews, always state these assumptions.

Suggested exercise: implement a `TimeMap`, remove its sorted-input assumption, and compare sorted insert `O(n)` with unsorted append + query-time sorting. Design tests for duplicate timestamps and missing keys.
