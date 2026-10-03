# The maths behind doom-esn

Status tags used throughout:

- **[D]** derived by hand here (check it yourself - that is the point of writing it out)
- **[T]** also checked numerically by a named test in `tests/`
- **[U]** unverified: a hypothesis or heuristic. Nothing in the code depends on a [U] claim
  being true; where one motivates a design choice it is called out.

## 1. Why the agent needs a state at all

Doom is a partially observed process: hidden state `s_t` (positions, health, ammo, monster
behaviour) and pixels `y_t = h(s_t)`. One frame is not Markov (a single image has no
velocity). A frame stack `(y_t, ..., y_{t-m+1})` is a *delay vector*, the object in Takens'
embedding theorem, which says that for a smooth autonomous system on a compact manifold of
dimension `d` and a generic observation function, `2d+1` delays suffice to recover the state.

**[U]** Doom violates the hypotheses (the agent injects inputs, there are discrete events such
as a monster dying, the dynamics are piecewise-smooth, not smooth). Takens is *motivation*
only: it explains why a memory of the recent past can stand in for the hidden state, not a
guarantee that it does here.

A reservoir replaces the hard window by a soft one:
`x_t = Phi(y_t, y_{t-1}, ...)`, a function of the whole past whose dependence on `y_{t-k}`
fades geometrically (section 3). Only the readout from `x_t` to action is learned.

## 2. The reservoir is a discretised ODE (the link to `cd_dynamax`)

`cd_dynamax` models continuous dynamics observed at irregular times, using the transition
`F_k = exp(A * Delta_k)` over each gap. The reservoir uses the same idea for its own memory:

```
tau * dx/dt = -x + c ,   c = tanh(W x_t + W_in u_t + b)   held constant on [t, t+Delta]
```

**[D][T]** With `c` constant the linear ODE solves exactly:

```
x(t+Delta) = c + (x_t - c) * exp(-Delta/tau) = (1-a) x_t + a c ,   a = 1 - exp(-Delta/tau)
```

Consequences:

- `a` is computed from the *actual* frame gap, so a variable frame-skip (`info["dt"]`) is exact
  for the leak part. Fixed-leak ESNs silently change their time-constant when you change the
  frame-skip. Test: `test_step_equals_exact_ode_solution_under_zero_order_hold`.
- Semigroup: two gaps `d1, d2` compose to one gap `d1+d2` (when there is no drive):
  `(1-a(d1))(1-a(d2)) = exp(-d1/tau) exp(-d2/tau) = 1 - a(d1+d2)`.
  Test: `test_leak_semigroup_property`.
- **[U-approximation]** Freezing the argument of `tanh` over the gap (zero-order hold) is exact
  only when `W = 0`; for `W != 0` it is a first-order-accurate step in `Delta/tau`.

## 3. Fading memory and the echo-state property

Take two reservoir states `x, y` driven by the same input. Since `|tanh'| <= 1`:

```
||x' - y'|| <= (1-a)||x - y|| + a||W(x - y)|| <= [ (1-a) + a ||W||_2 ] ||x - y||
```

**[D][T]** So `||W||_2 < 1` is *sufficient* for the map to be a contraction: any two initial
states merge (`test_contractive_reservoir_forgets_initial_condition`).

`rho(W) <= ||W||_2`, so the common rule "scale to `rho < 1`" is **not** sufficient in
general **[D]**. In practice `rho` slightly above 1 works because the *input saturates the
tanh*, which lowers the local gain. The honest quantity is the local linearisation

```
J_t = (1-a) I + a * diag(1 - tanh^2(pre_t)) * W
```

and the **conditional Lyapunov exponent** `lambda = lim (1/T) sum_t log ||J_t v_t||`
(tangent vector `v_t`, renormalised). `lambda < 0` means perturbations fade along the
trajectory the reservoir actually follows. At the fixed point (`x=0, u=0, b=0`) `J` is
constant and `lambda -> log rho((1-a)I + aW)`
**[D][T]** (`test_lyapunov_matches_log_spectral_radius_of_linearisation`).

**Memory horizon** `h = -1/lambda` steps: a perturbation decays like `exp(-k/h)`.
**[U]** as a statement about *useful memory*: it bounds sensitivity to old frames, but
sensitivity is not capacity.

## 4. How rich is the state? Effective rank

Let `C = Cov(x)` with eigenvalues `l_i >= 0` and `p_i = l_i / sum_j l_j`. Define

```
erank = exp( -sum p_i log p_i )          (Roy-Vetterli, Renyi order 1)
PR    = 1 / sum p_i^2 = exp(H_2)         (participation ratio, Renyi order 2)
```

**[D]** Renyi entropies are non-increasing in their order, `H_2 <= H_1`, hence
`1 <= PR <= erank <= n`. Equality `erank = PR = n` for a flat spectrum, `= 1` for rank one
(`test_effective_rank_and_pr_extremes`).

A reservoir of `n = 400` units does not give a 400-dimensional feature space. Frames of a
mostly static scene have a low-dimensional variation, and the calibration table in the run
output shows `erank` of order tens, `PR` of order ten, and condition numbers `1e4 - 1e6`.

## 5. Loss blindness

The training signal here is episode return, optimised through a readout
`z = Theta [x; 1]`. Four precise statements about what that signal cannot see.

**5.1 Structural blindness [D].** `x_t` does not depend on the learned parameters, so
`dL/dW = dL/dW_in = 0` identically. If the reservoir is collapsed (low `erank`), the training
signal cannot detect or repair it. Hence the reservoir is chosen *before* learning, by a
health gate on data only (`calibrate.py`), not by reward.

**5.2 Row-space lemma [D][T].** For softmax cross-entropy (the same holds for any
loss that touches `Theta` only through the logits) with `p_i = softmax(Theta x~_i)`:

```
dL/dTheta = (1/N) sum_i (p_i - e_{y_i}) x~_i^T       (x~ = [x; 1])
```

Every row lies in `span{x~_i}`. So gradient steps never move `Theta` in the null space of the
visited states (`test_readout_gradient_has_no_component_in_the_state_null_space`; the formula
itself is checked against finite differences).

**5.3 Null-space lemma [D][T].** Writing `Theta -> Theta + Delta` with each row of `Delta` in
`null(Cov(x))` changes every logit by the same amount for all states of an action
(`Delta mu`), so it can only shift the per-action bias; logit *differences* across states are
unchanged (`test_null_space_weights_only_shift_a_constant_per_action`). The number of
directions that change behaviour is therefore at most `n_actions * (rank(C) + 1)`, not
`n_actions * (n + 1)`.

**5.4 Consequence for the optimiser [U].** Gradient-free search (CMA-ES) feels the same
geometry as flat or near-flat directions: budget spent there is wasted. Diagonal CMA-ES also
assumes roughly separable coordinates, whereas reservoir features are strongly correlated
(`PR << n`). Both point to the same remedy: project the state onto the top eigenvectors of
`C` and rescale to unit variance (`z = Lambda_k^{-1/2} V_k^T (x - mu)`, so `Cov(z) = I_k`).
This is **derived but not implemented in v0.1**; whether it improves return is untested.

**5.5 What is *not* claimed.** Likelihood losses such as the one in `cd_dynamax` have other
blind spots (gauge invariance `(A,C) -> (TAT^-1, CT^-1)`; zero gradient at dead modes
`c_j b_j = 0`). They do not apply here because this project has no likelihood.

## 6. Fixed points, stability and saddles

**6.1 Where the fixed points are [D][T].** For a constant input `u` write `c = W_in u + b`.
`x = (1-a)x + a tanh(Wx + c)` is equivalent to `x = tanh(Wx + c)`: the leak cancels, so the
*location* of fixed points does not depend on `dt` or `tau`, but their *stability* does
(`test_fixed_point_ignores_leak_but_stability_does_not`).

**6.2 Classification [D][T].** At a fixed point `x*` the Jacobian is `J = (1-a) I + a D W`,
`D = diag(1 - x*^2)`, with eigenvalues `mu_i = 1 - a + a*lambda_i(DW)`. *Stable* if all
`|mu_i| < 1`; *saddle* if some but not all exceed 1; *unstable* if all do.

**6.3 The stability disc [D][T].** `|1 - a + a*lambda| < 1` is equivalent to
`|lambda + (1-a)/a| < 1/a`: a disc of radius `1/a` centred at `-(1-a)/a`, passing through
`lambda = 1`. For `a -> 1` it is the unit disc; for `a -> 0` it becomes the half-plane
`Re(lambda) < 1`, which is the continuous-time ODE condition. The unit disc lies inside it
(for `|lambda| <= 1`, `|lambda + c| <= 1 + c = 1/a`, equality only at `lambda = 1`), so
`rho(W) < 1` **is sufficient** for the origin to be locally stable at any leak, and not
necessary: the leak *widens* the region on the left half-plane. (This is a statement about the
fixed point, not the global echo-state property, which needs contraction; see section 3.)
Tests: `test_stability_disc_equals_modulus_test`, `test_unit_disc_is_inside_stability_disc...`.

**6.4 Pitchfork [D][T].** 1-D `x = tanh(beta x)`: for `beta <= 1` only the origin; for
`beta > 1` the origin loses stability and two stable points `+-x*` appear (`beta = 2`:
`x* = 0.9575`). In `n` dimensions this is the case where `m` eigenvalues of `DW` leave the
disc: the origin is a **saddle with `m` unstable directions**, and the system settles on
nonzero states. A bias or an input breaks the `x -> -x` symmetry (imperfect pitchfork), which
typically leaves one dominant attractor.

**6.5 What the data say [measured, `docs/results/data_shape*.json`].**
Reservoir `rho = 1.1`, 400 units, frame gap 4 tics:

- Origin of the *autonomous* map (b = 0): 5 of 400 eigenvalues outside the disc at `tau = 4`
  or 8, 3 at `tau = 16`; the disc test agrees with `|mu| > 1` for every eigenvalue.
  So the origin *is* a saddle (5 unstable directions): `rho = 1.1` is above the origin's
  stability limit.
- Blank input (autonomous with bias): 20 of 20 random starts converged to **stable** fixed
  points (0 saddles), conditional Lyapunov exponent -0.057 (weak contraction, ~17-step memory).
  Calling it a saddle-driven system would be wrong: the saddle is at the origin, the dynamics
  live elsewhere.
- Driven by 120 real frames held constant: 115 solved, all stable, none saddle (5 not found;
  not found is not the same as absent), median `max|mu|` 0.76, max 0.85; one distinct fixed
  point over 25 random starts at one input (no multistability seen).
- Along a real trajectory: local `max|mu|` median 0.78, max 0.89, **0 of 145 sampled steps
  expanding**. The input drives units into saturation (10% have `|x| > 0.9`), which shrinks
  `D` and hence the local gain below the autonomous value.

**6.6 What to do if a saddle is found.** In order of cost: (i) report the count of unstable
directions (`n_unstable_at_origin`, one eigendecomposition); (ii) confirm on *data* that the
driven map is still contracting (Lyapunov exponent < 0, local radius < 1; the calibration
gate already does this); (iii) if not, move the spectrum into the disc (lower `rho`, or raise
`a` by lowering `tau`) or raise the input scale. Multiple stable fixed points would be a
hysteresis memory and should be *reported*, not removed.

## 7. Saddles and plateaus in the learning problem

**7.1 No smooth saddles, but plateaus and cliffs [D].** With `a = argmax(Theta x)` the return
is piecewise constant in `Theta`: zero gradient almost everywhere, jumps at decision
boundaries. Gradient-free search is therefore the right tool, not a compromise. ES optimises
the Gaussian-smoothed `J_s(Theta) = E R(Theta + s*eps)`, whose gradient is `E[R eps]/s`; that
smoothed surface can have saddles (negative Hessian eigenvalue), and the population noise is
what lets the search leave them. In this project the observed problem was a **plateau**, below.

**7.2 Measuring the plateau [D][measured].** Model the return of candidate `c` on episode `e`
as `R_ce = mu + alpha_c + beta_e + eps_ce`. Candidates share episode seeds (common random
numbers), so `beta_e` cancels in a ranking and only `sigma_eps` matters. Mean squares from the
two-way table: `sigma_alpha^2 = (MS_cand - MS_resid)/n_e`. The reliability of a `k`-episode
mean for *ranking* candidates is `rho_k = sigma_alpha^2 / (sigma_alpha^2 + sigma_eps^2/k)`;
`k = sigma_eps^2/sigma_alpha^2` episodes give `rho = 1/2`.

Measured on the initial population (episodes are bimodal: ~19% positive, median -305):

| design | `sigma_alpha` | `sigma_eps` | `F` | `rho_3` | episodes for `rho = 0.5` |
|--------|--------------:|------------:|----:|--------:|-------------------------:|
| v0.1: raw 400-d reservoir, `s = 0.05` | 7.1 | 154 | 1.03 | 0.006 | ~469 |
| whitened reservoir, `s = 0.5` (pilot) | 45.5 | 147 | 1.96 | 0.22 | ~10 |
| whitened 2-window, `s = 0.5` (pilot; flawed arm, see 7.4) | 26.9 | 132 | 1.42 | 0.11 | ~24 |

(`F` near 1 means no detectable difference between candidates; 12 candidates x 10 episodes,
so the variance components are themselves noisy: treat the improvement as an order of
magnitude, not a precise factor.) **The v0.1 search was ranking noise at initialisation**, which is
consistent with one seed learning and the other not (not proof of the cause). My first noise estimate treated the shared
episode effect as noise and was wrong; the table above is the corrected analysis.

**7.3 Why whitening helps [D].** For an iid perturbation `Delta Theta ~ N(0, s^2)` the logit
change has variance `E||Delta theta_a . x||^2 = s^2 tr(C)` per action, with `C = Cov(x)`. The
perturbation therefore changes behaviour in proportion to `s^2 tr(C)`, and a raw reservoir
spends 400 coordinates on a state with participation ratio ~10 (section 4). Projecting onto the
top `k` axes and rescaling to `C = I_k` makes `tr(C) = k` known, removes the null directions of
5.2-5.3, and shrinks the search space from 1604 to 100 or fewer parameters.

**7.4 A design trap [measured reasoning].** A delay window must be built from *per-frame*
projected coordinates. PCA of two stacked frames keeps the high-variance common content and
discards the low-variance difference (velocity). `FeatureObservation(input_proj=...)` projects
first, then windows, then whitens jointly (which rescales the small velocity axis to unit
variance).

## 8. Probability: what state does the data want?

**8.1 Predictive state [D, standard].** For jointly Gaussian past `p` and future `f`, the
canonical correlations `rho_1 >= rho_2 >= ...` between them (singular values of
`S_pp^{-1/2} S_pf S_ff^{-1/2}`, the normalised Hankel operator) determine the minimal
state: its dimension is the number of non-negligible `rho_i` (credence Prop 5, Hankel rank),
and the predictive information is `I = -(1/2) sum log(1 - rho_i^2)` nats.

**8.2 Measured [`data_shape2.json`].** Random play, per-frame PCA to 12 axes, past 4 frames
vs future 4 frames, 3648 pairs, noise floor from shuffling the future (top `rho` 0.23):

```
rho = .997 .992 .988 .980 .962 .928 | .817 .813 .791 | .503 .458 .432 ...   (18 above the floor)
I(k) = 2.5, 4.6, 6.5, 8.1 (k=4), 10.4 (k=6), 11.5 (k=8), 12.3 (k=12) nats
```

The largest drop is between the 9th (0.79) and 10th (0.50) correlation: **the data want a
Markov state of about 6-9 dimensions**; 8 axes already carry 93% of the 12-axis predictive
information. Caveats: actions are not modelled, the PCA was fit on the same data, and random
play is not the policy's own distribution.

**8.3 How much memory [measured].** Ridge next-frame prediction (12 PCA axes, held-out
episodes), skill = 1 - MSE/Var: persistence 0.36; window `W = 1` 0.67, `W = 2` 0.80,
`W = 4` 0.83, `W = 8` 0.83, `W = 16` 0.84; reservoir (400 units) 0.71. Memory pays at two frames
(velocity is a first difference) and is saturated by four (16 tics). The reservoir state is
*below* the two-frame window. Caveat: the windows use the denoised 12-axis input and the
reservoir sees all 768 inputs through a ridge readout, so this is not a perfectly matched
comparison; it is the reason the classical null is an arm in the comparison.

**8.4 The Gaussian-filter version (cd_dynamax) [D, not implemented].** Fitting a linear-Gaussian
state space model to the 12 axes gives the optimal linear memory: the filter mean is a sum of
exponentials with rates `eig(A(I - KC))` (credence Prop 10), and an irregular frame gap `D_k`
enters as `F_k = exp(A D_k)`. The data above say a 2-frame window already holds most of the
predictive information in a handful of dimensions, so this is the next step *if* the window
stops being enough on a harder scenario, not a prerequisite.

## 9. Claim ledger

| # | Claim | Status | Evidence |
|---|-------|--------|----------|
| 1 | Leaky update is the exact exponential integrator for a frozen drive | [D][T] | `test_step_equals_exact_ode...` |
| 2 | Variable frame gaps compose (semigroup) | [D][T] | `test_leak_semigroup_property` |
| 3 | `||W||_2 < 1` implies the echo-state property | [D][T] | `test_contractive_reservoir...` |
| 4 | `rho(W) < 1` does not imply contraction / the echo-state property (non-normal `W`) | [D] | follows from `rho <= ||W||_2`; but `rho < 1` *is* sufficient for stability of the origin at any leak (row 14) |
| 5 | Tangent-vector estimator returns `log rho(J)` at the fixed point | [D][T] | `test_lyapunov_matches...` |
| 6 | `1 <= PR <= erank <= n` | [D] | Renyi monotonicity |
| 7 | Gradient has no null-space component | [D][T] | `test_readout_gradient...` |
| 8 | Null-space weights only shift per-action bias | [D][T] | `test_null_space_weights...` |
| 9 | Memory horizon `-1/lambda` is the useful memory length | [U] | heuristic only |
| 10 | Frame stack ~ Takens delay vector justifies a state for Doom | [U] | hypotheses of Takens not met |
| 11 | Higher `erank` (health gate) predicts higher return | [U] | weakly suggestive, **not significant** (2 seeds x 30 episodes, one scenario); see README |
| 12 | Whitened/projected state gives the search usable signal at initialisation | [D][measured] | section 7.2 table (pilot, 12 candidates x 10 episodes). That it improves *return* is **not shown**: planned comparison 1 is INCONCLUSIVE (README) |
| 13 | "k=4" Hodge-resonance hypothesis governs required state dimension | [U] | **not used by this project**; see README |
| 14 | Stability region of the leaky map is the disc `|lambda + (1-a)/a| < 1/a`; `rho < 1` suffices for the origin | [D][T] | `test_stability_disc...`, `test_unit_disc_is_inside...` |
| 15 | Fixed-point location is independent of leak; stability is not | [D][T] | `test_fixed_point_ignores_leak...` |
| 16 | Origin is a saddle with `m` unstable directions when `m` eigenvalues of `W` leave the disc | [D][T][measured] | `test_origin_is_a_saddle...`; m = 5 of 400 at `rho = 1.1` |
| 17 | The driven reservoir on real Doom frames has no saddle and no expanding direction | [measured] | 115/115 stable, 0/145 expanding steps; one random policy-free dataset, one scenario |
| 18 | Candidate x episode ANOVA with common random numbers gives the ranking reliability | [D] | formula in 7.2; estimator not separately tested |
| 19 | The v0.1 search had reliability ~0 at initialisation | [measured] | `data_shape2.json`: `F` = 1.03, ~469 episodes needed |
| 20 | The data want a state of ~6-9 dimensions; 2-4 frames of memory | [measured] | canonical correlations, next-frame skill (8.2-8.3) |
| 21 | The reservoir beats the delay-window null | [U] | next-frame skill favours the window (0.71 vs 0.80); return comparison INCONCLUSIVE (exact permutation, Holm p = 1.0, n = 5 vs 5) |

## 10. Not used here (on purpose)

The Hodge-Laplacian / persistent-homology "k=4 resonance" hypothesis is a separate research
thread. Doom's dynamics are not quasi-periodic (piecewise-linear motion plus discrete events),
so there is no torus for it to describe, and the project does not rely on it. Keeping it out
of the ledger's load-bearing claims is deliberate.
