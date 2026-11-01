# Machine Learning Coding — Intuition, Math, Tests and Complexity

The goal is to explain and implement the basic operations **without relying on sklearn model implementations**, while still using NumPy/PyTorch arrays for numerical operations. These are teaching models; production training adds regularization, data pipelines, validation and deployment controls.

## Start with the derivations

**New:** [Gradients from first principles — full chain-rule walkthrough](GRADIENTS_FROM_FIRST_PRINCIPLES.md) derives $\partial L/\partial w$, $\partial L/\partial b$, logistic regression gradients, the two-layer MLP backward pass, softmax cross-entropy, LayerNorm and LoRA updates. Each step includes shapes, intuition, and numerical checks in [runnable Python](../examples/gradient_walkthrough.py). The formulas use GitHub-rendered LaTeX.

For complete line-by-line derivations of the remaining notebook topics—including the K-means centroid derivative, distance-vectorization identity, LayerNorm input gradient, attention backward pass and LoRA adapter gradients—read [Mathematical Walkthroughs](ML_MATH_DEEP_DIVE.md).

## 1. Linear regression — NumPy

- **Intuition:** find the line/plane that minimizes average squared prediction error. A feature's coefficient tells how much the prediction changes for one unit of that feature (holding others fixed).
- **Model:** `y_hat = X @ w + b`; mean squared error `mean((y_hat-y)**2)`.
- **Gradient:** `dw = (2/N) X.T @ (y_hat-y)`; `db = 2 mean(y_hat-y)`.
- **Two implementations:** [`linear_regression_gd`](../src/interview_patterns/ml_from_scratch.py) iteratively updates parameters; [`linear_regression_lstsq`](../src/interview_patterns/ml_from_scratch.py) uses a numerically safer least-squares solver than explicitly inverting `X.T @ X`.
- **Complexity:** gradient descent `O(T*N*D)` time; additional `O(N+D)` transient memory. Dense least squares is typically `O(N*D² + D³)` when `N≥D` (solver dependent).
- **Interview traps:** forgetting bias, averaging gradient over N, wrong matrix dimensions, unscaled features causing divergence, multicollinearity/near-singular matrices.

## 2. Logistic regression — NumPy

- **Intuition:** classify by mapping a real-valued score to a probability; still a linear decision boundary in feature space.
- **Model:** `sigmoid(X @ w + b)` for binary classification; optimize binary cross-entropy rather than MSE.
- **Stable loss:** `mean(logaddexp(0, z) - y*z)`; avoids unstable `log(sigmoid(z))` and `exp(large)`.
- **Gradient:** `dw = X.T @ (sigmoid(z)-y)/N`; `db = mean(sigmoid(z)-y)`.
- **Complexity:** `O(T*N*D)` time, `O(N+D)` additional space.
- **Pitfalls:** probability calibration is not guaranteed; class imbalance, leakage, threshold selection, and correctly separated train/test data are separate concerns.

## 3. K-means — NumPy

- **Intuition:** alternate between assigning each point to its closest center and moving each center to its assigned points' mean.
- **Vectorization:** broadcast `X[:,None,:] - centers[None,:,:]` to an `[N,K,D]` temporary, then reduce over `D`.
- **Complexity:** `O(T*N*K*D)` time, `O(N*K + K*D)` additional peak materialized representations (plus broadcast temporary potentially `O(N*K*D)`).
- **Pitfalls:** initialization sensitivity and local optima, empty clusters, feature scaling, mismatch between Euclidean clusters and real data geometry. The included function recovers empty clusters deterministically but does not guarantee a global optimum.

## 4. k-nearest neighbors — NumPy

- **Intuition:** compare a query to labeled training samples, then vote among the K closest. No parameter training, but inference can be expensive.
- **Vectorization:** squared distance is `||a||² + ||b||² − 2 a·b`; negative rounding artifacts are clamped to zero.
- **Complexity:** all-query brute force about `O(Q*N*D + Q*N)` time and `O(Q*N)` additional memory for the distance matrix; use batching for large N/Q.
- **Pitfalls:** scaling, high-dimensional distances, ties, choosing K, leakage in standardization, duplicate points.

## 5. Two-layer MLP with manual backpropagation — NumPy

- **Intuition:** linear model + nonlinearity + linear model. The chain rule sends each output error backward through both layers.
- **Forward:** `Z1 = XW1+b1`, `H=tanh(Z1)`, `Z2=HW2+b2`, `P=sigmoid(Z2)`.
- **Backward:** `dZ2=(P-y)/N`, `dW2=H.T@dZ2`, `dH=dZ2@W2.T`, `dZ1=dH*(1-H²)`, `dW1=X.T@dZ1`.
- **Complexity:** `O(N*D*H)` time per full-batch step for H hidden units and a single output; auxiliary `O(N*H + D*H)` activations/gradients.
- **Proof:** [`test_manual_mlp_backprop_gradient`](../tests/test_ml_from_scratch.py) compares a parameter's analytical gradient with symmetric finite differences.
- **Pitfalls:** matrix transposes, gradient averaging, hidden activation derivative, in-place mutation while computing gradients, overfitting XOR-sized datasets.

## 6. LayerNorm — NumPy

- **Intuition:** normalize each observation *across its hidden features* (last axis); learnable `gamma` and `beta` restore flexibility.
- **Equation:** `(x - mean_last_axis(x)) / sqrt(var_last_axis(x)+eps) * gamma + beta`.
- **Complexity:** `O(number of elements)` time and memory for output and intermediates.
- **Contrast with BatchNorm:** BatchNorm typically uses minibatch statistics and running estimates at inference; LayerNorm uses current sample's chosen feature axes in both modes.
- **Test:** compare numerical result to PyTorch `torch.nn.functional.layer_norm`.

## 7. Multi-head self-attention — PyTorch

- **Intuition:** each token produces a query (what I need), key (what I offer), and value (what I contain). Multiple heads can learn different relationships.
- **Shapes:** `[B,L,D]` -> projections -> `[B,H,L,d]` with `D=H*d`. Attention scores: `[B,H,L,L]`.
- **Scaling:** divide QKᵀ by `sqrt(d)` to keep logits from becoming excessively sharp.
- **Causal mask:** prevent attending to positions strictly to the right; independently test that changing future tokens does not change earlier outputs.
- **Naive complexity:** `O(B*L*D² + B*L²*D)` time and `O(B*H*L²)` attention-score memory. Flash/SDPA implementations can avoid explicitly materializing the score matrix.
- **Pitfalls:** incorrect reshape/transposes, wrong mask convention, mixed precision overflow, comparing attention to LSTM complexity without specifying dimensions.

## 8. LoRA — PyTorch

- **Intuition:** freeze a pretrained weight matrix W and learn a low-rank update `ΔW = B @ A`, with rank much smaller than input/output dimensions.
- **Extra parameters:** `r*(d_in + d_out)` per adapted linear layer; not all model weights are necessarily frozen by default unless explicitly configured.
- **Complexity:** extra forward operations on the order of `O(batch*r*(d_in+d_out))`; adaptation reduces trainable weights, not necessarily all training activation memory.
- **Practice:** [`lora_toy.py`](../src/interview_patterns/lora_toy.py) verifies that base weights stay frozen and only adapters are trained.
- **Pitfalls:** treating LoRA as quantization, assuming lower trainable parameters guarantees equal quality, ignoring adapter targeting and rank choices.

## 9. BPE tokenization — Python

- **Intuition:** begin with small symbols, repeatedly merge frequent adjacent pairs, then apply those merges in the learned order.
- **Important distinction:** [`WordBPETokenizer`](../src/interview_patterns/tokenization.py) is a compact **word-level pedagogical** implementation with explicit word-boundary markers; it is *not* GPT-style byte-level BPE, does not include modern pretokenizer behavior, and is not a drop-in inference tokenizer.
- **Complexity:** naive training repeatedly scans corpus symbol pairs, approximately `O(M*corpus_symbols)` across M merges if symbol count stays bounded; actual costs depend on data structure. Encoding applies learned merges per word.
- **Pitfalls:** Unicode normalization, unknown symbols, merge-order consistency, vocabulary/merges compatibility, train/inference tokenizer mismatch.

## Additional workbook drills

| Drill | Code | Interview focus |
|---|---|---|
| Broadcasting/shape validation | `numerics.pairwise_squared_distances` | Align `[N,1,D]` with `[1,M,D]` |
| Pairwise cosine similarity | `numerics.pairwise_cosine_similarity` | Zero-vector policy; denominator epsilon |
| Stable softmax/logsumexp | `numerics.softmax_stable`, `numerics.logsumexp_stable` | Subtract maximum; test large logits |
| Rectangular image prefix sums | `numerics.prefix_2d`, `numerics.region_sum` | O(1) query after O(HW) preprocessing |
| Gradient checking | `torch_drills.gradient_check` | Autograd vs finite differences |
| Train/eval with holdout | `torch_drills.tiny_train_eval` | Seeds, `model.train` vs `eval`, checkpoints |
| Attention masks/memory | `attention.MultiHeadSelfAttention`, `torch_drills.attention_shape_demo` | Causality, O(L²) materialized scores |
| AMP/profiling | `torch_drills.amp_profile_demo` | Precision trade-offs, bottleneck analysis |

## Suggested whiteboard rehearsal

1. Write stable sigmoid and BCE from memory; explain why plain `exp(-z)` can overflow.
2. Derive logistic regression gradients and implement batch gradient descent in 15 minutes.
3. Show tensor shapes throughout a multi-head attention layer, then implement a causal mask.
4. Backpropagate through a two-layer MLP without autograd; validate one gradient numerically.
5. Design an experiment comparing nearest-neighbor and logistic regression baselines. Include split strategy, metrics and failure analysis.

**Source for topic selection (not copied code):** [MLE Coding Interview Guide, pinned revision](https://github.com/jyotipmahes/ML-Interview-Prep/blob/69e22abd95b4ed2d6d184b651b1b8af27f09448b/MLE_Coding_Interview_Guide.ipynb).
