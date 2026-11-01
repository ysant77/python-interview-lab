# Machine Learning Coding: Mathematical Walkthroughs

This is a **concept-to-code companion** for nine frequently used machine-learning coding exercises. It complements [Gradients from First Principles](GRADIENTS_FROM_FIRST_PRINCIPLES.md), which derives the core linear/logistic/MLP gradients with concrete numerical examples.

All mathematics is written using GitHub's native `$...$` (inline) and `$$...$$` (display) syntax. The equations are meant to render in GitHub's Markdown viewer. Unless stated otherwise, vectors in NumPy examples are **rows** collected into $X\in\mathbb R^{N\times D}$, and losses are averaged over the $N$ observations.

## 0. The common mental model: the chain rule is a sensitivity budget

A computation has intermediate values. To change a weight, ask how that weight affects the final loss **through every intermediate**.

$$
 w\longrightarrow z=wx+b\longrightarrow\hat y=f(z)\longrightarrow L=\ell(\hat y,y)
$$

A tiny change $\Delta w$ changes $z$ by about $x\Delta w$, changes $\hat y$ by about $f'(z)x\Delta w$, and changes the loss by about $\ell'(\hat y)f'(z)x\Delta w$. Therefore:

$$
\frac{\partial L}{\partial w}=
\frac{\partial L}{\partial\hat y}
\frac{\partial\hat y}{\partial z}
\frac{\partial z}{\partial w}.
$$

**Interpretation:** the product tells us how sensitive the loss is to a small change in the weight. In several dimensions, each $\partial L/\partial w_j$ measures one coordinate's effect while others stay fixed. Moving opposite the gradient is a *local* descent heuristic; a large learning rate can still increase the loss.

## 1. Linear regression: why $X^\top r$ appears

### 1.1 Scalar derivative

Let $\hat y=wx+b$, $r=\hat y-y$ and $L=r^2$.

$$
\frac{\partial L}{\partial w}=
\underbrace{\frac{\partial L}{\partial r}}_{2r}
\underbrace{\frac{\partial r}{\partial\hat y}}_{1}
\underbrace{\frac{\partial\hat y}{\partial w}}_{x}
=2rx.
$$

Likewise $\partial L/\partial b=2r$ because $\partial\hat y/\partial b=1$. At $x=2$, $y=5$, $w=1$, $b=0$, we get $r=-3$ and $\partial L/\partial w=-12$: increasing $w$ a little raises the prediction toward $5$, initially improving the fit.

### 1.2 Batch and vectorization

With $r=Xw+b\mathbf1-y$ and $L=\frac1N\sum_i r_i^2$:

$$
\frac{\partial L}{\partial w_j}=
\frac2N\sum_{i=1}^{N}r_i x_{ij}
\quad\Longrightarrow\quad
\boxed{\nabla_w L=\frac2N X^\top r},
\qquad
\boxed{\frac{\partial L}{\partial b}=\frac2N\sum_i r_i}.
$$

The transpose is not a magic gradient operation; $(X^\top r)_j=\sum_i x_{ij}r_i$ aggregates how *feature $j$* contributed to every sample's error. Shapes: $X^\top:(D,N)$ multiplied by $r:(N,)$ gives $(D,)$.

### 1.3 Normal equation: deriving a least-squares solution

Augment the feature matrix with a bias column: $\widetilde X=[X\;\mathbf1]$, and collect weights/bias as $\theta$. For the convention $L=\frac1N\|\widetilde X\theta-y\|_2^2$:

$$
\nabla_\theta L=\frac2N\widetilde X^\top(\widetilde X\theta-y).
$$

A differentiable minimum satisfies $\nabla_\theta L=0$, so:

$$
\widetilde X^\top\widetilde X\theta=\widetilde X^\top y.
$$

If $\widetilde X^\top\widetilde X$ is invertible, then

$$
\theta^*=(\widetilde X^\top\widetilde X)^{-1}\widetilde X^\top y.
$$

**Caution:** this inverse is a useful *derivation*, not an implementation recommendation. Collinearity can make the matrix singular or ill-conditioned; `np.linalg.lstsq` uses more robust linear algebra. Work and storage depend on solver and matrix shape; a dense least-squares solve is substantially more expensive than one gradient step on large feature sets.

## 2. Logistic regression: where the surprisingly simple $p-y$ comes from

For a logit $z=w^\top x+b$ and predicted probability $p=\sigma(z)=(1+e^{-z})^{-1}$, use binary cross-entropy:

$$
\ell=-y\log p-(1-y)\log(1-p),\qquad y\in\{0,1\}.
$$

First differentiate **loss with respect to probability**:

$$
\frac{\partial\ell}{\partial p}=-\frac yp+\frac{1-y}{1-p}
=\frac{p-y}{p(1-p)}.
$$

The sigmoid slope is $\frac{dp}{dz}=p(1-p)$. Multiplying the two derivatives cancels the denominator:

$$
\boxed{\frac{\partial\ell}{\partial z}=p-y}.
$$

Now $\partial z/\partial w_j=x_j$, so for a batch:

$$
\boxed{\nabla_w L=\frac1N X^\top(p-y)},
\qquad
\boxed{\frac{\partial L}{\partial b}=\frac1N\sum_i(p_i-y_i)}.
$$

**Interpret the sign:** for $y=1$ and $p=0.2$, $p-y=-0.8$: changing weights to increase the correct logit will decrease the loss. For $y=0$ and $p=0.9$, $p-y=0.9$: the update should reduce that logit.

**Stable evaluation** works in logit space:

$$
\ell(z,y)=\log(1+e^z)-yz=\operatorname{softplus}(z)-yz.
$$

Use `np.logaddexp(0, z) - y*z` to avoid taking logarithms of probabilities rounded to exactly 0 or 1. The derivative is still $\sigma(z)-y$.

## 3. K-means: the centroid update is an optimization step

K-means is **not** normally trained with backpropagation. It alternates between a discrete assignment step and a continuous minimization.

Given assignments $c_i\in\{1,\ldots,K\}$ and centroids $\mu_k\in\mathbb R^D$, minimize:

$$
J=\sum_{i=1}^N\|x_i-\mu_{c_i}\|_2^2.
$$

**Step A:** holding centroids fixed, choose the nearest centroid for each $x_i$. This is an `argmin`, so assignments jump discretely and are not smoothly differentiable in the usual way.

**Step B:** holding assignments fixed, look only at cluster $k$, containing $n_k$ points:

$$
J_k=\sum_{i:c_i=k}\sum_{j=1}^D (x_{ij}-\mu_{kj})^2.
$$

Differentiate with respect to **one** centroid coordinate $\mu_{kj}$:

$$
\frac{\partial J_k}{\partial \mu_{kj}}=
2\sum_{i:c_i=k}(\mu_{kj}-x_{ij})
=2\left(n_k\mu_{kj}-\sum_{i:c_i=k}x_{ij}\right).
$$

Set the derivative to zero:

$$
\boxed{\mu_k=\frac1{n_k}\sum_{i:c_i=k}x_i}.
$$

Hence the familiar **mean of assigned points** is not an arbitrary heuristic: it exactly minimizes squared distance for fixed, nonempty cluster assignments.

**Complexity:** for $T$ iterations, $O(TNKD)$ distance computations with straightforward dense arrays. Broadcasting `(X[:, None, :] - centers[None, :, :])` creates a potentially large $N\times K\times D$ intermediate; squared-norm identities can reduce peak memory. Empty clusters, local optima, and initialization remain important edge cases.

## 4. K-nearest neighbors: a distance formula, not a learned gradient

For query $q$ and training sample $x_i$, squared Euclidean distance is:

$$
\|q-x_i\|_2^2=(q-x_i)^\top(q-x_i)=\|q\|^2+\|x_i\|^2-2q^\top x_i.
$$

**Why useful?** Dot products can compute distances for many queries simultaneously:

```python
# Q: (M,D), X: (N,D)
dist2 = (Q * Q).sum(axis=1)[:, None] + (X * X).sum(axis=1)[None, :] - 2 * Q @ X.T
dist2 = np.maximum(dist2, 0.0)  # numerical roundoff guard
```

There is no $\partial L/\partial w$ in ordinary nonparametric KNN because standard KNN stores the reference samples instead of optimizing weights. The `argpartition`/vote operations are also not ordinarily a differentiable training objective. Distinguish **an ML prediction algorithm** from **a gradient-trained estimator**.

**Complexity:** all-pairs brute-force distances use $O(MND)$ arithmetic and $O(MN)$ space to store the full distance matrix. Feature scaling and tie-breaking can change outcomes.

## 5. Two-layer MLP: backpropagation as repeatedly reusing local slopes

With batch size $N$, input dimension $D$, hidden width $H$, and one binary output:

$$
Z_1=XW_1+b_1,\quad A_1=\tanh(Z_1),\quad Z_2=A_1W_2+b_2,\quad P=\sigma(Z_2).
$$

Shapes are $X:(N,D)$, $W_1:(D,H)$, $A_1:(N,H)$, $W_2:(H,1)$ and $P:(N,1)$. Use mean binary cross-entropy.

**Output layer:** sigmoid + BCE simplifies to

$$
\delta_2=\frac{\partial L}{\partial Z_2}=\frac{P-Y}{N}.
$$

Because $Z_2=A_1W_2+b_2$, each output weight sees the sum of hidden activations multiplied by the output error:

$$
\boxed{\frac{\partial L}{\partial W_2}=A_1^\top\delta_2},\qquad
\boxed{\frac{\partial L}{\partial b_2}=\sum_i\delta_{2,i}}.
$$

**Hidden layer:** the output uses $A_1$ through $W_2$, giving $\partial L/\partial A_1=\delta_2W_2^\top$. Since $\frac{d}{dz}\tanh z=1-\tanh^2z$, we get

$$
\delta_1=(\delta_2W_2^\top)\odot(1-A_1^2),
$$

and therefore

$$
\boxed{\frac{\partial L}{\partial W_1}=X^\top\delta_1},\qquad
\boxed{\frac{\partial L}{\partial b_1}=\sum_i\delta_{1,i,:}}.
$$

**Key idea:** gradients through branches **add**; gradients through sequential transformations **multiply** by the local derivative/Jacobian. Divide by $N$ **once**, at the mean-loss output, not at every layer. See [the NumPy walkthrough](../examples/gradient_walkthrough.py), which checks every MLP parameter against numerical finite differences.

## 6. LayerNorm: why a feature depends on the other features

For one token/vector $x\in\mathbb R^D$, LayerNorm uses mean and variance **across that vector's $D$ features**:

$$
\mu=\frac1D\sum_jx_j,\quad
c_j=x_j-\mu,\quad
v=\frac1D\sum_jc_j^2,\quad
s=(v+\varepsilon)^{-1/2},\quad
\hat x_j=c_js,\quad
y_j=\gamma_j\hat x_j+\beta_j.
$$

Let $g_j=\partial L/\partial y_j$ be the upstream gradient. The parameter gradients are immediate:

$$
\frac{\partial L}{\partial\gamma_j}=g_j\hat x_j,\qquad
\frac{\partial L}{\partial\beta_j}=g_j.
$$

For the input gradient, write $u_j=g_j\gamma_j$ and follow **both** paths from $x_j$ to $\hat x$: through the centering step $c=x-\mu$ and the shared scale $s$.

1. Because $\mu$ depends on all features, $dc_j/dx_k=\mathbf1_{j=k}-1/D$.
2. Because the centered features sum to zero, $dv/dx_k=2c_k/D$.
3. Since $s=(v+\varepsilon)^{-1/2}$, $ds/dx_k=-(c_k/D)s^3$.
4. Since $\hat x_j=c_js$, multiply the two paths and sum contributions weighted by $u_j$.

Consequently:

$$
\frac{\partial L}{\partial x_k}=
\underbrace{s\left(u_k-\frac1D\sum_j u_j\right)}_{\text{centering path}}
-\underbrace{\frac{c_k s^3}{D}\sum_j u_jc_j}_{\text{variance path}}.
$$

Since $\hat x_j=c_js$, this becomes the compact result:

$$
\boxed{
\frac{\partial L}{\partial x_k}=s\left[u_k-\operatorname{mean}(u)-\hat x_k\operatorname{mean}(u\odot\hat x)\right]}.
$$

A common bug is to treat $\mu$ and $v$ as constants when differentiating; this omits the two cross-feature effects. Verify a small `float64` implementation using PyTorch autograd before optimizing.

## 7. Multi-head attention: shapes, softmax and gradients

For one attention head, use row-oriented token matrix $X\in\mathbb R^{T\times D}$ and projections:

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V,\qquad
S=\frac{QK^\top}{\sqrt{d_k}},\quad
A=\operatorname{softmax}(S),\quad
O=AV.
$$

Here $Q,K\in\mathbb R^{T\times d_k}$, $V\in\mathbb R^{T\times d_v}$, scores $S$ and weights $A$ have shape $(T,T)$, and output $O$ has shape $(T,d_v)$.

**Why $1/\sqrt{d_k}$?** Under a simplified assumption that query/key coordinates are independent, zero-mean and unit-variance, a dot product contains $d_k$ terms, so its variance grows proportional to $d_k$. Dividing by $\sqrt{d_k}$ controls logit scale and reduces excessive softmax saturation. Actual learned representations need not satisfy those assumptions exactly.

### 7.1 Where the softmax derivative comes from

For one score row $s$, define $a_i=e^{s_i}/\sum_j e^{s_j}$. The numerator and denominator both depend on scores. Differentiation gives:

$$
\frac{\partial a_i}{\partial s_j}=a_i(\mathbf1_{i=j}-a_j).
$$

Thus increasing one logit raises its own probability while lowering others (the row must still sum to 1).

### 7.2 Backprop through attention, step by step

Let $G_O=\partial L/\partial O$ be supplied by later layers. First propagate through $O=AV$:

$$
\boxed{G_A=G_O V^\top},\qquad \boxed{G_V=A^\top G_O}.
$$

For each row $i$, the softmax Jacobian applied to upstream gradient $G_{A,i}$ simplifies to:

$$
\boxed{G_{S,i}=A_i\odot\left(G_{A,i}-\sum_j A_{ij}G_{A,ij}\right)}.
$$

Then propagate through the scaled dot product $S=QK^\top/\sqrt{d_k}$:

$$
\boxed{G_Q=\frac{G_SK}{\sqrt{d_k}}},\qquad
\boxed{G_K=\frac{G_S^\top Q}{\sqrt{d_k}}}.
$$

Finally, the projection-weight gradients are:

$$
\frac{\partial L}{\partial W_Q}=X^\top G_Q,\quad
\frac{\partial L}{\partial W_K}=X^\top G_K,\quad
\frac{\partial L}{\partial W_V}=X^\top G_V.
$$

For multi-head attention, reshape into head dimensions, compute per-head attention, concatenate and project; account for the final projection's gradient as another linear layer. A **causal mask** prevents a token from accessing future keys by assigning them effectively $-\infty$ before softmax, so their probabilities are zero. Masking all keys in a row needs special care to avoid `NaN`.

**Complexity:** naive self-attention projections typically cost $O(TD^2)$ and score/value products cost $O(T^2D)$ per batch (assuming standard head dimensions), with $O(T^2)$ score memory per head. Flash-style kernels optimize memory access and avoid materializing all score weights.

## 8. LoRA: differentiate through two smaller matrices

Consider frozen base weight $W_0\in\mathbb R^{d_{out}\times d_{in}}$ and trainable matrices $A\in\mathbb R^{r\times d_{in}}$, $B\in\mathbb R^{d_{out}\times r}$:

$$
W_{\mathrm{eff}}=W_0+sBA,\qquad s=\frac\alpha r.
$$

With batch-oriented $X\in\mathbb R^{N\times d_{in}}$, the output is

$$
Y=XW_0^\top+sXA^\top B^\top.
$$

Suppose the downstream loss supplies gradient $G=\partial L/\partial Y$ of shape $(N,d_{out})$. Define the conceptual full-weight gradient $M=G^\top X$ (shape $(d_{out},d_{in})$). Use the chain rule on the product $BA$:

$$
\boxed{\frac{\partial L}{\partial B}=sMA^\top},\qquad
\boxed{\frac{\partial L}{\partial A}=sB^\top M}.
$$

**Why these transposes?** The gradient for $B$ must have the same shape as $B$ ($d_{out}\times r$), while that for $A$ must have shape $r\times d_{in}$. Their matrix products make exactly those shapes.

**Important nuance:** the mathematical derivative for $W_0$ is $M$, but the base weight remains **frozen**, meaning the optimizer will not update it. If $B$ starts at zero, then initially $\partial L/\partial A=0$ while $\partial L/\partial B$ can be nonzero; $A$ begins receiving gradients after $B$ moves away from zero. This is expected behavior, not necessarily a bug.

Trainable adapter size is $r(d_{in}+d_{out})$ versus $d_{in}d_{out}$ full matrix entries. LoRA reduces optimized parameter count but does not automatically eliminate activation memory or guarantee equal accuracy.

## 9. BPE: why there is no loss derivative in the standard algorithm

A simple byte-pair encoding (BPE) trainer starts with basic symbols, counts adjacent symbol pairs, merges the most frequent pair, and repeats. The objective of a basic pedagogical BPE procedure is to **build a deterministic merge vocabulary**, not to minimize a differentiable scalar loss by gradient descent.

Example corpus (symbol boundaries shown for illustration):

```text
l o w _
l o w _
l o w e r _
```

The pair `l o` is frequent and could be merged into `lo`; later merges depend on the updated symbolization. The choice is an integer-count and ranking operation, not $\partial L/\partial w$. Different BPE variants tokenize bytes versus characters, use different pretokenizers, and handle boundaries differently. The repository's word-based tokenizer is intentionally pedagogical rather than byte-level tokenizer compatibility.

**Complexity:** a naive trainer rescans all tokenized symbols for each merge, approximately $O(MS)$ across $M$ merges and $S$ total symbol positions, ignoring bookkeeping differences; optimized implementations can maintain pair-frequency data structures.

## 10. How to verify mathematics in a coding interview

1. Write the **forward pass** before attempting any gradient.
2. Write each array's **shape**; the backward gradient must match the shape of its parameter.
3. Identify whether the loss uses `sum`, `mean`, or a $\tfrac12$ factor. Those choices change gradient coefficients.
4. Backpropagate through **one operation at a time**, reusing saved forward values.
5. Use numerical central differences on a tiny example when differentiability and precision permit:

$$
\frac{\partial L}{\partial\theta_j}\approx
\frac{L(\theta_j+\varepsilon)-L(\theta_j-\varepsilon)}{2\varepsilon}.
$$

6. Explain why this is a *testing oracle* rather than a scalable training algorithm: two loss evaluations per scalar parameter are expensive.

**Run examples:**

```bash
python examples/gradient_walkthrough.py
pytest tests/test_gradients.py tests/test_ml_from_scratch.py -q
```

**Reading path:** [ML coding implementation guide](ML_CODING_GUIDE.md) → [Gradients from First Principles](GRADIENTS_FROM_FIRST_PRINCIPLES.md) → this deeper mathematical companion → [Python reference implementations](../src/interview_patterns/ml_from_scratch.py).

**Reference for the exercise selection only:** [MLE Coding Interview Guide, pinned revision](https://github.com/jyotipmahes/ML-Interview-Prep/blob/69e22abd95b4ed2d6d184b651b1b8af27f09448b/MLE_Coding_Interview_Guide.ipynb). The explanations and code here are independently written.
