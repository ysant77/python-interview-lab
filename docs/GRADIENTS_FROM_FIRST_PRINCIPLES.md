# Gradients from First Principles: Why the Updates Work

> **Goal:** derive the gradients you would write in a Python interview, understand what each term means, and verify them numerically. This is about general ML training, not any particular application domain.
>
> **Rendering:** this file uses GitHub-supported Markdown mathematics: `$...$` for inline expressions and `$$...$$` for display equations. View the rendered `.md` on GitHub (rather than its raw source) to see the formulas.

## 0. The five-minute calculus refresher

A **derivative** measures how changing one number changes another. If $L(w)$ is the loss, then

$$
\frac{dL}{dw}\;\approx\;\frac{L(w+\varepsilon)-L(w-\varepsilon)}{2\varepsilon}
$$

for sufficiently small $\varepsilon$. A **partial derivative** $\partial L/\partial w_j$ asks how the loss changes when we vary *only* weight $w_j$ and hold all other weights fixed.

Gradient descent takes a step in the opposite direction to the local slope:

$$
 w_j \leftarrow w_j - \eta\frac{\partial L}{\partial w_j},
$$

where $\eta>0$ is the learning rate. Negative slope? Increasing that weight tends to lower the loss. Positive slope? Decreasing it tends to lower the loss. This is a *local*, not a global, guarantee.

**The chain rule** is the entire trick. For a pipeline $w\to z\to \hat y\to L$,

$$
\frac{\partial L}{\partial w}
=\frac{\partial L}{\partial \hat y}
 \frac{\partial \hat y}{\partial z}
 \frac{\partial z}{\partial w}.
$$

You can interpret each factor as a conversion: "loss per prediction" × "prediction per score" × "score per weight" = "loss per weight".

**Notation:** $N$ = number of samples; $D$ = input features; $H$ = hidden units; $X\in\mathbb{R}^{N\times D}$; $w\in\mathbb{R}^{D}$; $b\in\mathbb{R}$. The examples use the **mean** loss over all samples, so derivatives include a factor of $1/N$.

---

## 1. Linear regression + mean squared error

### 1.1 One sample, one parameter

Given a single feature $x$, weight $w$, bias $b$, and target $y$:

$$
\hat y = wx+b,\qquad L=(\hat y-y)^2.
$$

Work from the **outside** in. Let $r=\hat y-y$ denote the residual.

1. Squaring: $\displaystyle \frac{\partial L}{\partial r}=2r$.
2. Subtracting the target: $\displaystyle \frac{\partial r}{\partial \hat y}=1$.
3. Multiplying by weight: $\displaystyle \frac{\partial \hat y}{\partial w}=x$.
4. Bias contributes: $\displaystyle \frac{\partial \hat y}{\partial b}=1$.

Multiply the relevant slopes:

$$
\boxed{\frac{\partial L}{\partial w}=2(\hat y-y)x}
\qquad
\boxed{\frac{\partial L}{\partial b}=2(\hat y-y)}.
$$

**Concrete example:** $x=2$, $y=5$, $w=1$, $b=0$. The prediction is $2$, so $r=-3$, $L=9$, $\partial L/\partial w=-12$, and $\partial L/\partial b=-6$. A small **increase** in $w$ should reduce the error: $w\leftarrow1-\eta(-12)$. The sign makes intuitive sense.

### 1.2 Many samples, many features

For sample $i$ and feature $j$:

$$
\hat y_i = \sum_{j=1}^{D} x_{ij} w_j+b,
\qquad
L=\frac1N\sum_{i=1}^{N}(\hat y_i-y_i)^2.
$$

Each training sample contributes to the effect of weight $w_j$, so **sum across samples**:

$$
\begin{aligned}
\frac{\partial L}{\partial w_j}
 &= \frac1N\sum_{i=1}^N
    \underbrace{2(\hat y_i-y_i)}_{\text{loss slope}}
    \underbrace{x_{ij}}_{\text{feature's effect}}\\
 &=\frac2N\sum_{i=1}^N (\hat y_i-y_i)x_{ij},\\[4pt]
\frac{\partial L}{\partial b}
 &= \frac2N\sum_{i=1}^N(\hat y_i-y_i).
\end{aligned}
$$

Define the error vector $r=Xw+b\mathbf{1}-y$ (shape $N$). The vectorized form is

$$
\boxed{\nabla_w L = \frac{2}{N}X^\top r}\quad(D\text{ elements}),
\qquad
\boxed{\frac{\partial L}{\partial b}=2\,\operatorname{mean}(r)}.
$$

**Why the transpose?** $X^\top$ has shape $(D,N)$, $r$ has shape $(N,)$, and the result has one gradient for each of the $D$ features. `X.T @ residual` implements the sum over training samples, without a Python loop.

```python
residual = X @ w + b - y       # (N,)
d_w = (2 / len(X)) * (X.T @ residual)  # (D,)
d_b = 2 * residual.mean()               # scalar
w -= learning_rate * d_w
b -= learning_rate * d_b
```

**Complexity:** one full-batch gradient takes $O(ND)$ time; excluding input data, predictions and residuals take $O(N)$ storage and the weight gradient takes $O(D)$.

**Important convention:** If the loss is $L=\frac{1}{2N}\sum_i r_i^2$, the factor of 2 disappears. Both conventions are valid; don't mix their gradient formulas.

---

## 2. Logistic regression + binary cross-entropy

A linear model computes a **logit** $z=wx+b$. The sigmoid turns it into a probability:

$$
 p=\sigma(z)=\frac{1}{1+e^{-z}}.
$$

For binary label $y\in\{0,1\}$, binary cross-entropy is

$$
 L=-\left[y\log p+(1-y)\log(1-p)\right].
$$

We want $\partial L/\partial w$. Split the question into manageable derivatives.

**First: effect of probability on loss.**

$$
\frac{\partial L}{\partial p}
=-\frac{y}{p}+\frac{1-y}{1-p}
=\frac{p-y}{p(1-p)}.
$$

**Second: effect of logit on probability.** Sigmoid has the useful derivative

$$
\frac{\partial p}{\partial z}=p(1-p).
$$

**Third: effect of weight on logit.** $\partial z/\partial w=x$.

**Combine them: the middle factors cancel.**

$$
\frac{\partial L}{\partial z}
=\underbrace{\frac{p-y}{p(1-p)}}_{\partial L/\partial p}
\underbrace{p(1-p)}_{\partial p/\partial z}
=\boxed{p-y},
$$

$$
\boxed{\frac{\partial L}{\partial w}=(p-y)x},
\qquad
\boxed{\frac{\partial L}{\partial b}=p-y}.
$$

That simple result, $p-y$, is why sigmoid + binary cross-entropy is a useful pairing.

**Concrete example:** $x=2$, $y=1$, $w=b=0$: $z=0$, $p=0.5$. Then $\partial L/\partial w=(0.5-1)\times2=-1$. Increasing the weight increases the probability of the true positive class, reducing the loss locally.

For $N$ examples let $p=\sigma(Xw+b)$ and $e=p-y$:

$$
\boxed{\nabla_w L=\frac{1}{N}X^\top e},\qquad
\boxed{\frac{\partial L}{\partial b}=\frac1N\sum_i e_i}.
$$

```python
z = X @ w + b
p = sigmoid_stable(z)
e = p - y
d_w = X.T @ e / len(X)
d_b = e.mean()
```

**Numeric stability:** the algebra above describes the mathematics on $0<p<1$; in floating-point code, computing `log(p)` directly can produce `-inf` for saturated probabilities. Use the equivalent **logit-space** loss

$$
L=\operatorname{mean}\bigl(\operatorname{softplus}(z)-yz\bigr),
\quad \operatorname{softplus}(z)=\log(1+e^z),
$$

implemented with `np.logaddexp(0, z)`. The loss and gradient still correspond to the same binary cross-entropy objective.

**Complexity:** $O(ND)$ time and $O(N+D)$ auxiliary storage per full-batch iteration.

---

## 3. A two-layer neural network: manual backpropagation

The same chain rule applies, but the chain has more links. For $N$ samples, $D$ features and $H$ hidden units, define:

$$
\begin{aligned}
Z_1 &= XW_1+b_1 & &(N,H)\\
A_1 &= \tanh(Z_1) & &(N,H)\\
Z_2 &= A_1W_2+b_2 & &(N,1)\\
P   &= \sigma(Z_2) & &(N,1)
\end{aligned}
$$

with $W_1\in\mathbb{R}^{D\times H}$ and $W_2\in\mathbb{R}^{H\times 1}$. The loss is the **mean** binary cross-entropy over $N$ examples.

### 3.1 Start at the output

From the logistic regression derivation, the derivative of the mean loss with respect to final logits is

$$
\boxed{\delta_2=\frac{\partial L}{\partial Z_2}=\frac{P-Y}{N}} \quad (N,1).
$$

Then, using $Z_2=A_1W_2+b_2$,

$$
\boxed{\frac{\partial L}{\partial W_2}=A_1^\top\delta_2}\quad(H,1),
\qquad
\boxed{\frac{\partial L}{\partial b_2}=\sum_{i=1}^N\delta_{2,i}}\quad(1).
$$

**Intuition:** a hidden unit that is strongly active receives a bigger weight update if the output was wrong. `A1.T` sums that contribution over all samples.

### 3.2 Send the error into the hidden layer

Changing $A_1$ affects the output through $W_2$. The gradient flowing backwards is

$$
\frac{\partial L}{\partial A_1}=\delta_2W_2^\top\quad(N,H).
$$

But $A_1=\tanh(Z_1)$ and $\tanh'(z)=1-\tanh^2(z)=1-A_1^2$. Hence

$$
\boxed{\delta_1
=\frac{\partial L}{\partial Z_1}
=(\delta_2W_2^\top)\odot(1-A_1^2)}\quad(N,H),
$$

where $\odot$ denotes elementwise multiplication.

Finally, $Z_1=XW_1+b_1$ gives

$$
\boxed{\frac{\partial L}{\partial W_1}=X^\top\delta_1}\quad(D,H),
\qquad
\boxed{\frac{\partial L}{\partial b_1}=\sum_{i=1}^N\delta_{1,i,:}}\quad(H).
$$

### 3.3 Translate the entire derivation into Python

```python
z1 = X @ W1 + b1                # (N,H)
a1 = np.tanh(z1)                # (N,H)
z2 = a1 @ W2 + b2               # (N,1)
p = sigmoid_stable(z2)          # (N,1)

# One gradient per parameter; the division by N happens ONCE.
dz2 = (p - Y) / len(X)          # (N,1)
dW2 = a1.T @ dz2               # (H,1)
db2 = dz2.sum(axis=0)          # (1,)

dz1 = (dz2 @ W2.T) * (1 - a1**2)  # (N,H)
dW1 = X.T @ dz1                # (D,H)
db1 = dz1.sum(axis=0)          # (H,)
```

**Reverse chain in words:** prediction error → output logit → output weights and hidden activations → activation nonlinearity → hidden weights and input features.

**Traps:** (1) forgetting $1/N$; (2) applying it twice; (3) using $\tanh'(z)=1-z^2$ instead of $1-\tanh^2(z)$; (4) transposing the wrong matrix; (5) updating $W_2$ *before* using its original value to compute $\delta_1$.

**Complexity:** full-batch forward/backward is $O(NDH+NH)$ time for one hidden layer and a scalar output, simplified as $O(NDH)$ for $D,H\ge1$. Intermediate activations and gradient buffers take $O(NH+DH+N)$ auxiliary space.

---

## 4. Multi-class softmax + cross-entropy (useful extension)

For $C$ classes, logits $z_1,\ldots,z_C$ produce

$$
p_j=\frac{e^{z_j}}{\sum_{k=1}^C e^{z_k}},
\qquad
L=-\sum_{j=1}^C y_j\log p_j,
$$

where the target $y$ is one-hot and $\sum_j y_j=1$.

The softmax Jacobian is

$$
\frac{\partial p_k}{\partial z_j}=p_k(\mathbf{1}_{k=j}-p_j).
$$

Using $\partial L/\partial p_k=-y_k/p_k$ and summing across $k$:

$$
\begin{aligned}
\frac{\partial L}{\partial z_j}
 &=\sum_{k=1}^C \left(-\frac{y_k}{p_k}\right)
 p_k(\mathbf{1}_{k=j}-p_j)\\
 &=-y_j+p_j\sum_{k=1}^C y_k\\
 &=\boxed{p_j-y_j}.
\end{aligned}
$$

For a batch **mean** loss, divide by $N$. This generalizes the $p-y$ pattern you saw in binary logistic regression.

**Stable coding:** subtract the maximum logit before exponentiating; this does not change softmax probabilities, because numerator and denominator are scaled equally.

---

## 5. LayerNorm gradients (optional advanced topic)

For one vector $x\in\mathbb{R}^D$, set

$$
\mu=\frac1D\sum_jx_j,\quad
v=\frac1D\sum_j(x_j-\mu)^2,\quad
\hat x_j=\frac{x_j-\mu}{\sqrt{v+\varepsilon}},\quad
y_j=\gamma_j\hat x_j+\beta_j.
$$

If an upstream layer supplies $g_j=\partial L/\partial y_j$, parameter gradients are straightforward:

$$
\boxed{\frac{\partial L}{\partial \gamma_j}=g_j\hat x_j},\qquad
\boxed{\frac{\partial L}{\partial \beta_j}=g_j}.
$$

With $u=g\odot\gamma$ and $s=1/\sqrt{v+\varepsilon}$, the input gradient, accounting for the fact that *every* feature affects the mean and variance, is

$$
\boxed{\frac{\partial L}{\partial x}
=s\left[u-\operatorname{mean}(u)\mathbf1
-\hat x\operatorname{mean}(u\odot\hat x)\right]}.
$$

Every `mean` here is over the **normalized feature axis**. The epsilon remains inside the square root. Per sample, this takes $O(D)$ time and auxiliary storage. For a batch, sum $\partial L/\partial\gamma$ and $\partial L/\partial\beta$ over samples when those parameters are shared.

---

## 6. LoRA: derivatives when only small adapter matrices train

With a frozen base linear layer, one common column-vector notation is

$$
W_{\mathrm{eff}}=W_0+sBA,
\quad W_0\in\mathbb{R}^{d_{out}\times d_{in}},
\quad B\in\mathbb{R}^{d_{out}\times r},
\quad A\in\mathbb{R}^{r\times d_{in}}.
$$

Here $s=\alpha/r$ is the adapter scaling. In the row-vector batch convention, $Y=XW_{\mathrm{eff}}^\top$. Let $G=\partial L/\partial Y$ be the output gradient and $M=G^\top X$ the gradient *that the effective weight would receive*. Then

$$
\boxed{\frac{\partial L}{\partial B}=sMA^\top},
\qquad
\boxed{\frac{\partial L}{\partial A}=sB^\top M},
\qquad
\underbrace{\frac{\partial L}{\partial W_0}}_{\text{not updated}}=M.
$$

The frozen base may have a nonzero mathematical derivative; **frozen** means no optimizer update is applied to it. This distinction matters in interviews. Extra trainable parameters: $r(d_{in}+d_{out})$ instead of $d_{in}d_{out}$ for full weight adaptation (biases excluded).

---

## 7. Prove the gradient rather than trusting your algebra

A *finite-difference gradient check* independently estimates one partial derivative by modifying just one weight:

$$
 g_j^{\mathrm{numeric}}=\frac{L(w_j+\varepsilon)-L(w_j-\varepsilon)}{2\varepsilon}.
$$

Compare it against your analytical derivative. Both should be close (not necessarily identical due to floating-point rounding). Use small, well-behaved test inputs, a `float64` dtype, and $\varepsilon\approx10^{-5}$. Numeric gradients can be unreliable near nondifferentiable points or if the step is too small.

```bash
python examples/gradient_walkthrough.py
pytest tests/test_gradients.py -q
```

The example prints predictions, residuals, analytical partial derivatives, finite-difference estimates and gradient errors for linear regression, logistic regression and **every** parameter array of a two-layer MLP.

**Complexity warning:** analytical backprop computes *all* gradients in roughly the same order as a few forward passes. Finite differences require about **two loss evaluations per scalar parameter**, making them useful for tests, but impractical for large neural networks.

### Spoken interview summary

> "I first write the forward equations and check every tensor shape. I differentiate the loss with respect to its immediate output, then use the chain rule to move backward. For a batch-mean loss I divide exactly once by the batch size. I vectorize the sample sums using matrix multiplication, and verify a few weights using symmetric finite differences."

**Continue:** [ML coding guide](ML_CODING_GUIDE.md) · [worked example code](../examples/gradient_walkthrough.py) · [tests](../tests/test_gradients.py).
