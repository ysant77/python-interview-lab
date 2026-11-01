# Two-Week Focused Study Plan

This is a priority-based syllabus, **not a prediction of interview questions**. Adjust days to available time. Aim for 45–75 focused minutes/day; keep a second lightweight review block only if sustainable.

| Day | DSA (30–40 min) | ML/Python (20–35 min) | What to explain aloud |
|---|---|---|---|
| 1 | Two Sum (LC 1), valid anagram (242) | NumPy broadcast distances | Hash map invariant, tensor shapes |
| 2 | Longest unique substring (3), min subarray (209) | stable softmax/logsumexp | Window expansion/shrink invariant, overflow |
| 3 | Binary search (704), rotated search (33) | linear regression GD + least squares | Bounds and gradients |
| 4 | Valid parentheses (20), daily temperatures (739) | logistic regression | Stack invariants, cross-entropy |
| 5 | Merge intervals (56), insert interval (57) | 2D prefix sums | Sort order, off-by-one and inclusion-exclusion |
| 6 | Top K frequent (347), kth-largest (215) | k-NN and k-means | Heap vs sorting, distance memory |
| 7 | Number of islands (200), course schedule (207) | 2-layer MLP backprop | Graph representation, chain rule |
| 8 | Product except self (238), subarray sum (560) | LayerNorm vs BatchNorm | O(1) extra space vs output, axis choice |
| 9 | Linked list cycle (141), reverse list (206) | PyTorch autograd + gradient check | Pointer rewiring, detach and finite differences |
| 10 | Tree levels (102), tree diameter (543) | train/eval loop and holdout | DFS return values, leakage |
| 11 | DP stairs (70), robber (198), coin change (322) | multi-head attention | DP state, `[B,H,L,L]` scores |
| 12 | LRU Cache (146), TimeMap (981) | LoRA parameters and freezing | Amortized guarantees, what trains |
| 13 | Serialize strings (271), MinStack (155) | BPE merges + tokenizer edge cases | Invariants, Unicode and round-trip |
| 14 | Mock: 2 unseen coding questions | Mock: explain one ML algorithm | Real-time debugging, complexity, tests |

## Immediate priorities

1. **P0 algorithms:** hash maps, sliding windows, stacks, binary search, intervals, graphs, heaps; solve 1–2 problems each from `catalog/problems.csv`.
2. **P0 ML engineering:** NumPy broadcasting and stable softmax; PyTorch autograd, a train/eval loop and one manual gradient derivation.
3. **P1 expansions:** trees, greedy, DP, topological sort and 15 design problems; attention and LoRA.
4. **P2 after mocks:** specialized graphs/backtracking/heap combinations and unimplemented catalog links.

## Oral answer format (5–8 minutes)

State the problem and assumptions → show brute-force baseline → identify invariant/optimization → derive complexity → code → test normal/edge cases → discuss failure modes. For machine learning, add objective function, gradients/shapes, reproducibility, numerical stability, train/test split and metric choice.

## Short post-session log

Keep a personal file (not committed) with **problem URL, attempted date, independent solve time, failed tests, explanation gaps, and next review date**. Do not publish your interview schedules or private recruiter communications.
