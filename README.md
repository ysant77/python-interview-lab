# Python Interview Lab

**Learn the intuition. Implement the pattern. Prove it with tests. Explain the trade-offs.**

A public, Python-first reference lab for **data structures and algorithms (DSA), practical Python/API design, NumPy, machine learning from scratch, and PyTorch engineering**. Written for interview preparation and revision, not as a substitute for solving problems independently.

## What is in this repository?

| Track | What to practise | Start here |
|---|---|---|
| DSA patterns | 27 pattern families; windows, graphs, search, greedy, DP, heaps | [Pattern field guide](docs/DSA_PATTERNS.md) |
| Python design | 15 stateful design exercises: cache, queue, time-indexed store, serialization | [Design guide](docs/PYTHON_DESIGN.md) |
| NumPy foundations | Broadcasting, distances, cosine similarity, stable softmax, prefix sums | [`numerics.py`](src/interview_patterns/numerics.py) |
| ML from scratch | Linear/logistic regression, k-means, k-NN, a 2-layer MLP with manual gradients, LayerNorm | [ML coding guide](docs/ML_CODING_GUIDE.md) · [step-by-step gradient derivations](docs/GRADIENTS_FROM_FIRST_PRINCIPLES.md) |
| PyTorch | Autograd, train/eval, attention, gradient checks, AMP, LoRA | [`torch_drills.py`](src/interview_patterns/torch_drills.py), [`attention.py`](src/interview_patterns/attention.py) |
| Tokenization | Deterministic, pedagogical **word-level** BPE | [`tokenization.py`](src/interview_patterns/tokenization.py) |
| Problem index | 261 LeetCode links, priority, pattern, implementation status, source | [Solution index](docs/SOLUTION_INDEX.md) · [`catalog/problems.csv`](catalog/problems.csv) |
| Source chapter index | Additional topic headings and URL references from a public DSA patterns repository | [`catalog/source_chapters.csv`](catalog/source_chapters.csv) |

> **Accuracy of coverage:** All **178 unique workbook-indexed LeetCode IDs**, and all **261 indexed LeetCode problem links** across the supplied source materials, now have Python reference implementations. This is *code coverage*, not a claim of exhaustive correctness or individual test coverage. A catalog entry is not a solved problem. Check [current coverage](catalog/coverage.json) and the `status` column. Passing tests is not proof of correctness for all inputs. We welcome counterexamples and additional tests.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate               # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e '.[test]'
pytest -q
```

For NumPy and PyTorch tracks:

```bash
python -m pip install -e '.[ml]'
pytest -q
```

List problems by tier and status (built-in Python modules only):

```bash
python scripts/practice.py --tier P0 --limit 20
python scripts/practice.py --tier P1 --pattern "Binary Search"
python scripts/practice.py --status catalogued --limit 30
python scripts/practice.py --id 200
```

Try a problem **before** reading the solution:

```python
from interview_patterns.arrays import two_sum
assert two_sum([2, 7, 11, 15], 9) == [0, 1]

from interview_patterns.ml_from_scratch import linear_regression_lstsq
import numpy as np
x = np.arange(6, dtype=float).reshape(-1, 1)
y = 3 * x[:, 0] + 2
model = linear_regression_lstsq(x, y)
assert np.allclose(model.predict(x), y)
```

### Understand the derivatives, not just their formulas

Start with [**Gradients from First Principles**](docs/GRADIENTS_FROM_FIRST_PRINCIPLES.md) and continue into the [**nine-topic mathematical deep dive**](docs/ML_MATH_DEEP_DIVE.md) for worked mathematical derivations of $\partial L/\partial w$ and $\partial L/\partial b$, including logistic regression, MLP backpropagation, softmax cross-entropy, LayerNorm, and LoRA. The equations use GitHub-compatible LaTeX. Run the NumPy walkthrough to check analytical derivatives against numerical approximations:

```bash
python examples/gradient_walkthrough.py
pytest tests/test_gradients.py -q
```

## A recommended revision loop

1. Read one pattern's **invariant and when-to-use signal** in the [DSA guide](docs/DSA_PATTERNS.md).
2. Solve one linked problem independently in 20–35 minutes. Write brute force first if needed.
3. Explain time complexity, **auxiliary** space, mutation, tie-breaking and edge cases.
4. Inspect the Python implementation and run `pytest` or add a counterexample.
5. Re-solve it cold 2–3 days later. Record **what went wrong**, not merely whether you passed.

### Prioritization

- **Tier 1 — urgent:** hash maps, sliding windows, two pointers, binary search, stacks, intervals, heaps, graph BFS/DFS, prefix sums; plus NumPy broadcasting/stable softmax, manual gradients, minimal PyTorch train/eval.
- **Tier 2 — valuable:** linked lists, trees, topological sort, DP, data-structure APIs, attention masking, linear/logistic regression, LayerNorm, k-means/k-NN.
- **Tier 3 — later:** harder backtracking/graph variations, specialty heaps, knapsack variants, full catalogue completion, optimized parallel training.

See [the two-week study order](docs/STUDY_PLAN.md) and [spoken interview answer template](docs/INTERVIEW_ANSWER_TEMPLATE.md) for a focused schedule rather than attempting hundreds of problems at once.

## Repository conventions

Each **implemented** LeetCode solution has a Python function/class with an `LC <number>` docstring, an algorithm idea and approximate time/space complexity. Corresponding official problems are linked in the CSV catalog and in the [searchable solution index](docs/SOLUTION_INDEX.md). Multiple problems may share a pattern or an implementation technique; the CSV identifies the canonical reference.

- Code language: **Python only** (including NumPy/PyTorch).
- Format/data: Markdown, CSV, JSON and GitHub Actions YAML are documentation/metadata, not additional programming languages.
- Tests: `pytest`; deterministic seeds for examples; selected numerical gradient comparisons and PyTorch tests.
- Complexity: unless stated otherwise, additional space excludes input and explicitly required output; numerical operations' complexity depends on dimensions and implementation.
- Scope: educational, not a hardened production library; check assumptions and constraints before reuse.
- **No role-specific concepts, company names, private interview questions, employer details, or proprietary material** in this repository.

## References and attribution

The code here is written independently for study; source materials are referenced, **not copied**.

- [LeetCode problems](https://leetcode.com/problemset/) — individual problem URLs are in [`catalog/problems.csv`](catalog/problems.csv); some premium problems may require access.
- [DSA patterns reference](https://github.com/Chanda-Abdul/Several-Coding-Patterns-for-Solving-Data-Structures-and-Algorithms-Problems-during-Interviews) — source chapter taxonomy and publicly linked practice problems; see [`catalog/source_chapters.csv`](catalog/source_chapters.csv).
- [MLE Coding Interview Guide (pinned commit)](https://github.com/jyotipmahes/ML-Interview-Prep/blob/69e22abd95b4ed2d6d184b651b1b8af27f09448b/MLE_Coding_Interview_Guide.ipynb) — inspiration for the nine-topic ML coding track; implementations and tests in this lab are independent, not a verbatim redistribution of that notebook.
- [NumPy broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html), [PyTorch autograd](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html), [PyTorch scaled dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html).

## Contribution policy

PRs adding **a correct implementation + complexity + tests + a brief invariant/pitfall** are preferable to unchecked batches of links. Do not copy solutions from unlicensed third-party repositories. See [contribution checklist](docs/CONTRIBUTING.md). The original code in this repository is available under the [MIT License](LICENSE).
