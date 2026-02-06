# Adam: A Method for Stochastic Optimization

We introduce Adam, an algorithm for first-order gradient-based optimization of stochastic objective functions, based on adaptive estimates of lower-order moments. The method is straightforward to implement, is computationally efficient, has little memory requirements, is invariant to diagonal rescaling of the gradients, and is well suited for problems that are large in terms of data and/or parameters. The method is also appropriate for non-stationary objectives and problems with very noisy and/or sparse gradients. The hyper-parameters have intuitive interpretations and typically require little tuning. Some connections to related algorithms, on which Adam was inspired, are discussed. We also analyze the theoretical convergence properties of the algorithm and provide a regret bound on the convergence rate that is comparable to the best known results under the online convex optimization framework. Empirical results demonstrate that Adam works well in practice and compares favorably to other stochastic optimization methods. Finally, we discuss AdaMax, a variant of Adam based on the infinity norm.

## Implementation Details

### Brainstorming & Design Choices

To implement the core logic of the Adam paper, the primary design choice was to construct a custom optimizer class inheriting from `torch.optim.Optimizer`. While PyTorch provides a highly optimized C++ backend for Adam, using the high-level API would hide the mathematical intricacies described in the paper. By implementing `AdamFromScratch` in pure Python (utilizing PyTorch tensors for vectorization), we can explicitly map the code to the equations found in Kingma and Ba's work (2014).

**Trade-offs:**
*   **Efficiency vs. Clarity:** Our Python-loop implementation is slower than the native CUDA kernel version of `torch.optim.Adam`. However, it allows for a line-by-line correspondence with the academic theory, which is the goal of this exercise.
*   **Dataset:** We selected **MNIST** as the dataset. While the paper mentions large-scale problems, MNIST is the standard "Hello World" for optimization papers. It is large enough (60k images) to show stochastic noise behavior but small enough to train in seconds on Colab. It serves as a valid proxy for evaluating convergence speed.

### Dataset & Tools

*   **Dataset:** MNIST (Modified National Institute of Standards and Technology database).
*   **Source:** [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)
*   **Access Method:** Accessed via `torchvision.datasets.MNIST`, which handles downloading and parsing the binary files automatically.
*   **Tools:** PyTorch (Gradient computation), Matplotlib/Seaborn (Visualization).

### Theoretical Foundation

**1. The Problem Space: Stochastic Gradient Descent (SGD)**
In deep learning, we seek to minimize an objective function $f(\theta)$. In Stochastic Gradient Descent, we estimate the gradient using a mini-batch of data:
$$ \theta_{t} = \theta_{t-1} - \alpha \cdot g_t $$
where $g_t = \nabla_\theta f_t(\theta_{t-1})$.

SGD suffers from two major issues:
1.  **Ravines:** It struggles to navigate areas where the surface curves much more steeply in one dimension than another.
2.  **Learning Rate Sensitivity:** A single global learning rate $\alpha$ applies to all parameters, regardless of the sparsity or frequency of updates for that parameter.

**2. First Moment (Momentum)**
To solve oscillation in ravines, we use the history of gradients (momentum). We maintain an exponential moving average of the gradient $m_t$:
$$ m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t $$
This smooths out the noise in the gradient estimation.

**3. Second Moment (Adaptive Learning Rates)**
To handle sparse features and different scales, we look at the uncentered variance (second raw moment) of the gradients. This was inspired by RMSProp. We track $v_t$:
$$ v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 $$
where $g_t^2$ is the element-wise square of the gradient.

**4. The Adam Contribution: Bias Correction**
A key novelty in Adam (Adaptive Moment Estimation) compared to RMSProp is **Bias Correction**. Because $m_0$ and $v_0$ are initialized to vectors of zeros, the estimates are biased towards zero, especially during the initial time steps and when decay rates ($\beta_1, \beta_2$) are close to 1. Adam corrects this:
$$ \hat{m}_t = \frac{m_t}{1 - \beta_1^t} $$
$$ \hat{v}_t = \frac{v_t}{1 - \beta_2^t} $$

**5. The Update Rule**
The final parameter update combines these corrected moments:
$$ \theta_t = \theta_{t-1} - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} $$

### Implementation Walkthrough

The `AdamFromScratch` class implements Algorithm 1 from the paper:

1.  **Initialization (`__init__`)**: We store hyperparameters $\alpha$ (`lr`), $\beta_1, \beta_2$ (`betas`), and $\epsilon$ (`eps`).
2.  **State Management**: Inside `step()`, we check if `state` is empty. If so, we initialize `exp_avg` ($m_0$) and `exp_avg_sq` ($v_0$) as zero tensors matching the parameter shape.
3.  **Moment Updates**:
    *   `exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)` directly corresponds to $m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$.
    *   `exp_avg_sq` is updated similarly using $g_t^2$ (`grad` multiplied by itself via `addcmul_`).
4.  **Bias Correction**:
    *   We calculate $1 - \beta_1^t$ and $1 - \beta_2^t$ using the current step count `t`.
    *   We compute `m_hat` and `v_hat` by dividing the moving averages by these correction factors.
5.  **Parameter Update**:
    *   We compute the denominator: $\sqrt{\hat{v}_t} + \epsilon$.
    *   We perform the update: `p.data.addcdiv_`. This performs $\theta \leftarrow \theta - \eta \cdot \frac{\hat{m}}{\text{denom}}$.

### Expected Plots & Visuals

When running the notebook, you will see a comparison plot titled **"Convergence Comparison: Adam vs SGD on MNIST"**.

*   **X-axis:** Training steps (iterations).
*   **Y-axis:** Cross Entropy Loss.
*   **Interpretation:**
    *   The **Adam** curve (Blue) should descend rapidly in the first few epochs. This demonstrates Adam's ability to adapt learning rates for each parameter, allowing it to take larger effective steps where gradients are consistent.
    *   The **SGD** curve (Orange) will likely descend slower or require a carefully tuned learning rate to match Adam's initial speed. It serves as a baseline to validate that the custom Adam implementation is functionally superior for this un-tuned scenario.

## Verification & Testing

The code provides a faithful implementation of the Adam optimizer as described in the original paper (Kingma & Ba, 2014). 

**Strengths:**
- **Algorithm Fidelity:** The implementation correctly follows Algorithm 1 from the paper. It calculates the biased moments ($m_t$, $v_t$), performs the bias correction ($\\hat{m}_t$, $\\hat{v}_t$), and updates parameters using the correct step rule.
- **Input Validation:** The `__init__` method includes necessary checks for hyperparameters (learning rate, epsilon, betas).
- **State Management:** It correctly handles the initialization of moment states ($m_0, v_0$) and tracks the time step `t`.
- **Weight Decay:** The implementation applies weight decay as L2 regularization on the gradients before the moment updates, which is the standard approach for the original Adam algorithm (distinct from AdamW).

**Minor Observations:**
- **Memory Efficiency:** The explicit creation of temporary tensors `m_hat`, `v_hat`, and `denom` inside the loop is less memory-efficient than highly optimized libraries (like PyTorch's native C++ implementation) which often fuse operations, but it is excellent for readability and educational purposes.
- **Epsilon Placement:** The code adds epsilon *after* the square root of the bias-corrected second moment (`v_hat.sqrt().add_(eps)`). This matches the primary algorithm in the paper. Some variations add epsilon inside the square root or before bias correction, but this implementation adheres to the standard definition.

**Verdict:** The code is syntactically correct, logically sound, and functional.