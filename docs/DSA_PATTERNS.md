# DSA Pattern Field Guide

Use these 27 patterns as **recognition heuristics**, not templates to force onto every task. For each problem, explain the invariant and why it works.

The index includes the full shared workbook list; [problem catalog](../catalog/problems.csv) provides implementation links and source references.

## 1. Sliding Window — P0

**Intuition:** Maintain a contiguous range and shrink when the invariant is violated.

**Complexity (typical):** O(n) for monotone pointers; O(k) tracking memory.

**Pitfall:** Negative values can invalidate the usual sum-window strategy.

**Start with:** [LC 3](https://leetcode.com/problems/longest-substring-without-repeating-characters/) · [LC 209](https://leetcode.com/problems/minimum-size-subarray-sum/)

**Listed IDs:** 3, 76, 209, 424, 567, 904

## 2. Two Pointers — P0

**Intuition:** Move pointers based on a monotonic rule, often after sorting.

**Complexity (typical):** Often O(n); O(1) auxiliary after sorting.

**Pitfall:** Sorting may destroy original indices; duplicate handling matters.

**Start with:** [LC 11](https://leetcode.com/problems/container-with-most-water/) · [LC 167](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/)

**Listed IDs:** 11, 15, 16, 18, 42, 167

## 3. Fast/Slow Pointers (Linked List) — P1

**Intuition:** Move pointers at different speeds to detect cycles or find the middle.

**Complexity (typical):** O(n) time, O(1) space.

**Pitfall:** Check null pointers before two-step movement.

**Start with:** [LC 141](https://leetcode.com/problems/linked-list-cycle/) · [LC 876](https://leetcode.com/problems/middle-of-the-linked-list/)

**Listed IDs:** 141, 142, 19, 876, 160, 234

## 4. Binary Search on Sorted Data — P0

**Intuition:** Keep an invariant on the answer-bearing half of a sorted search domain.

**Complexity (typical):** O(log n) time, O(1) space.

**Pitfall:** Duplicate bounds and inclusive endpoints can change the result.

**Start with:** [LC 704](https://leetcode.com/problems/binary-search/) · [LC 33](https://leetcode.com/problems/search-in-rotated-sorted-array/)

**Listed IDs:** 33, 34, 35, 153, 162, 704

## 5. Binary Search on Answer — P1

**Intuition:** Binary-search the smallest/largest feasible numeric value.

**Complexity (typical):** O(feasibility(n)·log(range)).

**Pitfall:** Prove feasibility monotonicity and handle impossible inputs.

**Start with:** [LC 875](https://leetcode.com/problems/koko-eating-bananas/) · [LC 1011](https://leetcode.com/problems/capacity-to-ship-packages-within-d-days/)

**Listed IDs:** 875, 1011, 410, 774, 1283, 1482

## 6. Hashing / Frequency Maps — P0

**Intuition:** Trade memory for O(1) expected membership and counting.

**Complexity (typical):** Often O(n) time and space.

**Pitfall:** Collisions, negative values, ties and Unicode assumptions.

**Start with:** [LC 1](https://leetcode.com/problems/two-sum/) · [LC 242](https://leetcode.com/problems/valid-anagram/)

**Listed IDs:** 1, 49, 128, 217, 242, 347

## 7. Prefix Sum / Running Sum — P1

**Intuition:** Convert many range sums into differences of accumulated totals.

**Complexity (typical):** O(n) preprocessing; O(1) range query.

**Pitfall:** Use a sentinel 0 for clean off-by-one handling.

**Start with:** [LC 560](https://leetcode.com/problems/subarray-sum-equals-k/) · [LC 724](https://leetcode.com/problems/find-pivot-index/)

**Listed IDs:** 303, 560, 724, 930, 974, 523

## 8. Difference Array / Range Updates — P2

**Intuition:** Record only range boundary changes, then reconstruct by prefix sum.

**Complexity (typical):** O(n+updates) time, O(n) space.

**Pitfall:** Right boundary should usually subtract at end+1.

**Start with:** [LC 1094](https://leetcode.com/problems/car-pooling/) · [LC 1109](https://leetcode.com/problems/corporate-flight-bookings/)

**Listed IDs:** 370, 1094, 1109, 1893, 1943, 2381

## 9. Monotonic Stack — P0

**Intuition:** Discard earlier candidates that cannot be useful after seeing a new value.

**Complexity (typical):** O(n) time, O(n) space.

**Pitfall:** Choose strict vs non-strict comparison carefully.

**Start with:** [LC 739](https://leetcode.com/problems/daily-temperatures/) · [LC 496](https://leetcode.com/problems/next-greater-element-i/)

**Listed IDs:** 739, 496, 503, 84, 85, 901

## 10. Monotonic Queue / Deque — P2

**Intuition:** Maintain an ordered deque of best candidates within a rolling window.

**Complexity (typical):** O(n) amortized time, O(n) space.

**Pitfall:** Expire indices outside the window before using the head.

**Start with:** [LC 239](https://leetcode.com/problems/sliding-window-maximum/) · [LC 1438](https://leetcode.com/problems/longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit/)

**Listed IDs:** 239, 862, 1425, 1438, 1499, 1696

## 11. Heap / Top K — P0

**Intuition:** Maintain the k best candidates rather than sorting everything.

**Complexity (typical):** O(n log k) typical; O(k) heap space.

**Pitfall:** Tie breaking and heap direction are common errors.

**Start with:** [LC 347](https://leetcode.com/problems/top-k-frequent-elements/) · [LC 215](https://leetcode.com/problems/kth-largest-element-in-an-array/)

**Listed IDs:** 215, 347, 692, 703, 973, 1046

## 12. Intervals — P0

**Intuition:** Sort by endpoints, then merge or schedule by overlap.

**Complexity (typical):** O(n log n) time, O(n) output.

**Pitfall:** Clarify whether touching endpoints overlap.

**Start with:** [LC 56](https://leetcode.com/problems/merge-intervals/) · [LC 57](https://leetcode.com/problems/insert-interval/)

**Listed IDs:** 56, 57, 252, 253, 435, 452

## 13. Greedy Scheduling / Sorting — P1

**Intuition:** Prove a local choice never blocks an optimal global outcome.

**Complexity (typical):** Often O(n log n) sorting.

**Pitfall:** Greedy intuition alone is not a proof.

**Start with:** [LC 55](https://leetcode.com/problems/jump-game/) · [LC 621](https://leetcode.com/problems/task-scheduler/)

**Listed IDs:** 45, 55, 406, 621, 763, 134

## 14. Linked List Manipulation — P1

**Intuition:** Rewire nodes using sentinels to simplify boundary cases.

**Complexity (typical):** Often O(n) time, O(1) space.

**Pitfall:** Save next pointers before mutation.

**Start with:** [LC 21](https://leetcode.com/problems/merge-two-sorted-lists/) · [LC 24](https://leetcode.com/problems/swap-nodes-in-pairs/)

**Listed IDs:** 21, 23, 24, 25, 92, 138

## 15. Tree DFS — P1

**Intuition:** Let each recursive call compute a reusable subtree property.

**Complexity (typical):** O(n) time, O(height) call stack.

**Pitfall:** Differentiate the global optimum from the value returned to a parent.

**Start with:** [LC 104](https://leetcode.com/problems/maximum-depth-of-binary-tree/) · [LC 543](https://leetcode.com/problems/diameter-of-binary-tree/)

**Listed IDs:** 104, 112, 113, 543, 124, 226

## 16. Tree BFS / Level Order — P1

**Intuition:** Use a queue and process one level at a time.

**Complexity (typical):** O(n) time, O(width) queue.

**Pitfall:** Snapshot the current queue length before enqueuing children.

**Start with:** [LC 102](https://leetcode.com/problems/binary-tree-level-order-traversal/) · [LC 199](https://leetcode.com/problems/binary-tree-right-side-view/)

**Listed IDs:** 102, 103, 199, 515, 637, 116

## 17. BST Problems — P2

**Intuition:** Exploit ordered left/right subtrees to restrict search.

**Complexity (typical):** O(height) typical, worst O(n).

**Pitfall:** BST may be unbalanced; deletion has several cases.

**Start with:** [LC 98](https://leetcode.com/problems/validate-binary-search-tree/) · [LC 700](https://leetcode.com/problems/search-in-a-binary-search-tree/)

**Listed IDs:** 98, 99, 230, 235, 450, 700

## 18. Backtracking Basics — P1

**Intuition:** Build a partial candidate, recurse, then undo the change.

**Complexity (typical):** Usually exponential; O(depth) auxiliary.

**Pitfall:** Skip duplicate permutations correctly.

**Start with:** [LC 78](https://leetcode.com/problems/subsets/) · [LC 46](https://leetcode.com/problems/permutations/)

**Listed IDs:** 46, 47, 77, 78, 90, 39

## 19. Backtracking with Constraints — P2

**Intuition:** Prune search branches immediately after a constraint fails.

**Complexity (typical):** Usually exponential, task-specific.

**Pitfall:** Undo visited flags even on early returns.

**Start with:** [LC 79](https://leetcode.com/problems/word-search/) · [LC 40](https://leetcode.com/problems/combination-sum-ii/)

**Listed IDs:** 40, 17, 79, 131, 51, 52

## 20. Graph BFS / DFS — P0

**Intuition:** Explore connected vertices/cells with visited bookkeeping.

**Complexity (typical):** O(V+E) time, O(V) space.

**Pitfall:** Choose 4-way vs 8-way grid adjacency and mutation policy.

**Start with:** [LC 200](https://leetcode.com/problems/number-of-islands/) · [LC 733](https://leetcode.com/problems/flood-fill/)

**Listed IDs:** 200, 695, 733, 994, 1091, 1254

## 21. Topological Sort / DAG — P1

**Intuition:** Use incoming degrees and process zero-indegree nodes.

**Complexity (typical):** O(V+E) time, O(V+E) space.

**Pitfall:** Cycle detection requires checking how many nodes were processed.

**Start with:** [LC 207](https://leetcode.com/problems/course-schedule/) · [LC 210](https://leetcode.com/problems/course-schedule-ii/)

**Listed IDs:** 207, 210, 802, 1462, 1203, 2115

## 22. Union Find / DSU — P2

**Intuition:** Track connected components with find/union, path compression and rank/size.

**Complexity (typical):** Amortized almost O(1) per operation.

**Pitfall:** Off-by-one labels and already-connected unions.

**Start with:** [LC 547](https://leetcode.com/problems/number-of-provinces/) · [LC 684](https://leetcode.com/problems/redundant-connection/)

**Listed IDs:** 547, 684, 1319, 1579, 990, 1202

## 23. Shortest Path — P2

**Intuition:** Use BFS for unweighted edges or Dijkstra for nonnegative weighted paths.

**Complexity (typical):** BFS O(V+E); Dijkstra O((V+E) log V).

**Pitfall:** Dijkstra fails with negative edges; constrained stops may need Bellman-Ford.

**Start with:** [LC 743](https://leetcode.com/problems/network-delay-time/) · [LC 787](https://leetcode.com/problems/cheapest-flights-within-k-stops/)

**Listed IDs:** 743, 787, 1514, 1631, 1334, 1976

## 24. MST / Graph Greedy — P2

**Intuition:** Connect all vertices at minimum total edge cost (Prim/Kruskal).

**Complexity (typical):** Typically O(E log E).

**Pitfall:** MST cost and bottleneck path objectives are related but not identical.

**Start with:** [LC 1584](https://leetcode.com/problems/min-cost-to-connect-all-points/) · [LC 778](https://leetcode.com/problems/swim-in-rising-water/)

**Listed IDs:** 1584, 1135, 1168, 1489, 778, 1102

## 25. Trie — P2

**Intuition:** Store a character-by-character prefix tree for incremental matching.

**Complexity (typical):** Insert/search O(word length), memory ~ total characters.

**Pitfall:** Wildcard search may branch exponentially; include terminal marker.

**Start with:** [LC 208](https://leetcode.com/problems/implement-trie-prefix-tree/) · [LC 211](https://leetcode.com/problems/design-add-and-search-words-data-structure/)

**Listed IDs:** 208, 211, 212, 648, 677, 1268

## 26. Bit Manipulation — P2

**Intuition:** Use identities with XOR, shifts, masks, and bit counts.

**Complexity (typical):** Often O(1) per fixed-width integer.

**Pitfall:** Signed fixed-width behavior differs from Python unbounded ints.

**Start with:** [LC 136](https://leetcode.com/problems/single-number/) · [LC 191](https://leetcode.com/problems/number-of-1-bits/)

**Listed IDs:** 136, 137, 191, 338, 268, 190

## 27. 1D DP Basics — P1

**Intuition:** Define the state, recurrence, base cases and evaluation order.

**Complexity (typical):** Often O(n) or O(n·amount).

**Pitfall:** Check what transitions require the previous vs current row.

**Start with:** [LC 70](https://leetcode.com/problems/climbing-stairs/) · [LC 198](https://leetcode.com/problems/house-robber/)

**Listed IDs:** 70, 198, 213, 322, 279, 300

