# PPO Learning for Univariate Bézier Homotopy Paths (Research Report)

## 1. Objective

In the homotopy continuation method for univariate polynomials, we define the path in coefficient space by a **Bézier curve** and optimize its **interior control points** via reinforcement learning (PPO). The goals are to increase the **success rate** of path tracking, reduce the **number of steps** (accepted and rejected steps) required for tracking, and, if possible, shorten computation time. After training, we compare the linear homotopy path (conventional linear interpolation) and the Bézier path on the same problem and tracker settings to evaluate effectiveness.

---

## 2. Problem Setting (Univariate Only)

Consider a univariate polynomial of degree $n$ with complex coefficients. The polynomial is written as
$$
f(x) = c_n x^n + c_{n-1} x^{n-1} + \cdots + c_0, \quad c_i \in \mathbb{C} \;\;(i = 0, \ldots, n)
$$
with $n+1$ coefficients ($c_i$ is the coefficient of $x^i$ in $f$, and $\mathbb{C}$ is the set of complex numbers).

The start polynomial is fixed as $g(x) = x^n - 1$ (the univariate total-degree start system). Its coefficient vector is $\mathbf{c}_g \in \mathbb{C}^{n+1}$. The target polynomial is $f$, and its coefficients $c_0, \ldots, c_n$ are sampled per episode. For each $i = 0, \ldots, n$, the real part $\Re(c_i)$ and imaginary part $\Im(c_i)$ are given by
$$
\Re(c_i), \, \Im(c_i) \overset{\text{i.i.d.}}{\sim} \mathcal{U}[-b, b]
$$
($b > 0$ is a parameter, e.g. $b = 5$). The resulting coefficient vector is $\mathbf{c}_f \in \mathbb{C}^{n+1}$. The endpoints of the Bézier curve are $\mathbf{c}_g$ (start) and $\mathbf{c}_f$ (target).

---

## 3. Bézier Path and Control Points

Let the Bézier degree be $d \geq 2$ (e.g. 2, 3, 4) and the control points be $P_0, P_1, \ldots, P_d \in \mathbb{C}^{n+1}$. With endpoints $P_0 = \mathbf{c}_g$ and $P_d = \mathbf{c}_f$, the degree-$d$ Bézier curve for parameter $t \in [0, 1]$ is defined by
$$
B(t) = \sum_{i=0}^{d} \binom{d}{i} t^i (1-t)^{d-i} \, P_i \in \mathbb{C}^{n+1}
$$
so that $B(0) = P_0$, $B(1) = P_d$, giving a smooth path in coefficient space $\mathbb{C}^{n+1}$ from $\mathbf{c}_g$ to $\mathbf{c}_f$.

In this work, the **interior control points** $P_1, \ldots, P_{d-1}$ are the quantities to optimize. For each $k \in \{1, 2, \ldots, d-1\}$, we take the linear interpolation $\bar{P}_k = (1 - k/d) P_0 + (k/d) P_d$ as a baseline and define $P_k$ by a perturbation $\boldsymbol{\delta}_k \in \mathbb{C}^{n+1}$:
$$
P_k = \bar{P}_k + \boldsymbol{\delta}_k
$$
A latent dimension $m$ (hyperparameter) is introduced, and the agent’s action is $\mathbf{a} \in \mathbb{R}^{(d-1)m}$. From $\mathbf{a}$, the perturbations $\boldsymbol{\delta}_1, \ldots, \boldsymbol{\delta}_{d-1}$ are determined.

---

## 4. Episode and Reward (One-Step)

The episode length is one step ($T=1$). Each episode fixes one problem $(g, f)$; the agent outputs an action once to set the interior control points, the Bézier path is evaluated once, and the episode ends.

- **State $\mathbf{s}$**  
  $\mathbf{s} = (\mathbf{c}_f, \mathbf{c}_g)$.

- **Action $\mathbf{a}$**  
  $\mathbf{a} = (\mathbf{a}_1, \ldots, \mathbf{a}_{d-1})$ with each $\mathbf{a}_k \in \mathbb{R}^m$. A fixed linear map $\Phi : \mathbb{R}^m \to \mathbb{C}^{n+1}$ gives the perturbation $\boldsymbol{\delta}_k = \Phi(\mathbf{a}_k)$. Components of $\mathbf{a}$ are clipped to a finite interval (e.g. $[-1,1]$) when used.

- **Reward $r$**  
  After running the tracker, let $N_{\text{acc}}$ be the number of accepted steps and $N_{\text{rej}}$ the number of rejected steps. The tracking cost is $J = N_{\text{acc}} + \rho N_{\text{rej}}$ on success ($\rho \geq 0$ is a weight) and $J = M$ (constant penalty) on failure. Let $J_{\text{linear}}$ be the cost when tracking the linear homotopy path (linear interpolation from $\mathbf{c}_g$ to $\mathbf{c}_f$) with the same tracker. Since there is only one step, the reward is a terminal bonus:
  $$
  r = c_{\text{linear}} \cdot (J_{\text{linear}} - J)
  $$
  ($c_{\text{linear}} > 0$ is a coefficient). Reducing $J$ with the Bézier path relative to the linear path is thus rewarded positively.

---

## 5. Learning Method

The algorithm uses PPO (Proximal Policy Optimization) with the standard continuous-action PPO. The environment is a custom RL environment; observations include target/start coefficients and progress-related information. Parameters (degree $n$, Bézier degree $d$, latent dimension $m$, reward coefficients, tracker parameters, etc.) are set per experiment.

---

## 6. Experimental Setup

We use degree $n \in \{5, 10, 20, 40, 80\}$, Bézier degree $d \in \{3, 4\}$, and episode length $T=1$. The reward gives a terminal bonus equal to the cost difference from the linear baseline scaled by $c_{\text{linear}}=10$, with failure penalty $M$ and tracking-cost weight $\rho=1$. Tracker step-control parameters ($\alpha$, $\beta_{\omega_p}$, $\beta_\tau$, strict $\beta_\tau$, etc.) are set in line with prior work. PPO total steps, rollout length, learning rate, discount $\gamma$, GAE $\lambda$, etc. are tuned in the usual ranges. Evaluation is done periodically on fixed validation instances, with comparison to the linear path and the zero-perturbation ($z=0$) Bézier path.

---

## 7. Results

An example of training and evaluation is shown in the figure below.

<p align="center">
    <img src="img/result_bezier_univar_ppo.png" alt="Bézier univariate PPO learning results" width="700">
</p>

The success rate is 1.0 under all conditions, so the comparison reduces to **efficiency (\#steps)**. Overall, the Bézier path reduces \#steps for many degrees, and **$d_b=4$** is consistently best.

In terms of average efficiency (mean, median), $d_b=4$ is roughly minimal for degree $=$ 10, 20, 40, 80, yielding about **15–19\%** reduction versus the linear path (see improvement rates in the table). For stability (standard deviation), $d_b=4$ also minimizes std for many degrees, suggesting reduced variance and more stable tracking. For the best case (min), $d_b=4$ is minimal at degree $=$ 10, 20, 40, 80, so even easy instances are not degraded. For the worst case (max), at degree $=$ 10, 20, 80, $d_b=4$ minimizes max as well (worst-case improvement), whereas at degree $=$ 5 and degree $=$ 40 the linear path has the smallest max, so curve paths can slightly worsen the worst instances (the degree $=$ 40 gap is small).

For $d_b=3$, mean/median improve but max worsens at degree $=$ 5, 10, 40, suggesting “good on average but prone to outliers.” The higher degrees of freedom with $d_b=4$ likely allow the path to avoid difficult regions and reduce rejected steps, thus lowering \#steps.

**Practical conclusion:** we recommend **$d_b=4$** by default (stable improvement at high degree).
