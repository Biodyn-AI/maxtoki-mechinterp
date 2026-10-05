# Results: something can be trained on Norman, and what it learns is saturation

`code/learn_interaction.py`, output `outputs/results.json`. Posing the interaction per gene gives
115 pairs x 1,000 genes = **115,000 training examples**. Whole pairs are held out (5-fold over pairs),
so the double of an evaluation pair is never seen in training.

**Ceiling: I-profile split-half 0.395 -> full-data reliability 0.566 -> a perfect model reaches 0.752.**

| model | corr(pred I, true I) | 95% CI | variance of I explained | share of ceiling |
|---|---|---|---|---|
| additive (I = 0) | 0.000 | - | 0% | 0% |
| **sum-only** (linear in dA+dB) | **+0.370** | [+0.303, +0.438] | **18.4%** | 49% |
| saturation (+product, min, max, sign) | +0.377 | [+0.309, +0.442] | 19.7% | 50% |
| saturation + per-gene offset | **+0.406** | [+0.347, +0.462] | 21.1% | **54%** |

**1. A trained model works.** Sum-only reaches +0.370 against an additive baseline of exactly 0, with a
tight interval, on held-out pairs. The best model reaches +0.406, **54% of the ceiling**.

**2. But one feature does most of the work.** Sum-only, a single number (how large dA+dB is), gets
0.370, which is 91% of the best model. Products, minima, maxima and sign agreement add **+0.007**.
Per-gene offsets add another +0.029. **The interaction term is mostly shrinkage: when the two singles
together predict a large change, the real change is smaller.** This is a ceiling/floor effect. It
needs no regulatory network and does not depend on which genes were perturbed.

**3. Most of the interaction is not pair-specific.** It is a generic non-linearity. **Any method that
claims to predict interactions must report the sum-only baseline. Beating "additive" shows nothing.**

**4. Real headroom remains: 0.406 against a ceiling of 0.752.** The best model reaches 54% of what
is achievable, so **just under half of the reliable interaction signal (46%) is unexplained** by
saturation and per-gene offsets. That residual is the best-defined open target here: it is
reproducible, it is not explained by a trivial baseline, and nothing tested predicts it.
