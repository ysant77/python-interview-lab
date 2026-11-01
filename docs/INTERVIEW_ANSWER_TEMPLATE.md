# How to Present a Solution in a Real Interview

A correct solution is necessary but insufficient. Interviewers also assess whether you identify constraints, prove the invariant, choose meaningful tests, and communicate time/space trade-offs.

## Standard answer structure (6–10 minutes)

1. **Restate and constrain:** input size, ordering, duplicates, negative numbers, null inputs, whether mutation is acceptable.
2. **Baseline:** give a simple algorithm and explain why it is correct; estimate Big-O.
3. **Recognize pattern:** state the *invariant* that permits an improvement.
4. **Walk through:** manually trace a normal case and one counterexample to a common mistake.
5. **Implement:** code Python with named variables and testable helpers.
6. **Evaluate:** time, auxiliary space, worst vs average, recursion stack, numeric stability and data copying.
7. **Test:** empty, singleton, duplicates, out-of-range, overflow/precision, tie behavior.

## Example A — LC 3: longest substring without repeating characters

**Brute force:** enumerate all substrings and check unique characters (up to O(n³) with repeated rescans).

**Invariant:** window `[left, right]` contains no repeated character. Record the last seen index of each character; when a duplicate enters the window, move `left` to one past its previous occurrence, but never move left backward.

**Complexity:** O(n) time, O(min(n, distinct alphabet)) auxiliary space. Edge cases: empty string, all same characters, Unicode characters.

**Code:** [`longest_unique_substring`](../src/interview_patterns/arrays.py).

## Example B — LC 875: binary search on the answer

**Question:** choose the smallest eating speed meeting a time deadline. Directly searching each possible speed takes O(n*max(piles)).

**Invariant:** if speed `s` meets the deadline, every speed greater than `s` also meets it. This monotone feasibility property enables binary search.

**Complexity:** O(n log(max(piles))) time, O(1) space. Test a tight deadline, one pile, and a boundary case where feasibility flips by one.

**Code:** [`min_eating_speed`](../src/interview_patterns/arrays.py).

## Example C — stable NumPy softmax

**Naive approach:** `np.exp(logits)/np.sum(np.exp(logits))` overflows when logits are very positive and may underflow when all are very negative.

**Identity:** subtract the maximum per row or requested axis before exponentiation; softmax is unchanged by adding the same constant to every logit in a row.

**Complexity:** O(number of logits) time and additional memory; specify the reduction axis and shape of output. Test `[-1000,-999,-998]` and `[1000,1001,1002]`.

**Code:** [`softmax_stable`](../src/interview_patterns/numerics.py).

## Example D — two-layer MLP without autograd

**Forward:** `H = tanh(XW1+b1)`, `P = sigmoid(HW2+b2)`.

**Backward:** use chain rule: start with the BCE derivative `P-y`, propagate through `W2`, then through `1-H²` to `W1`. Average over batch N exactly once.

**Tests:** compare a selected analytical gradient with central finite difference before attempting to train. Investigate a large discrepancy before adjusting learning rate.

**Complexity:** O(N*D*H) operations per pass for single-output and H hidden units; memory dominated by activations and gradients.

**Code:** [`mlp_loss_gradients`](../src/interview_patterns/ml_from_scratch.py).
