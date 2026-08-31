Let's build this from the underlying motivation, since the mechanism is genuinely elegant once you see _why_ it's built this way.
## The problem: the big, accurate model is slow — one token at a time

The best-quality language model (the "target" model) is typically large and expensive — generating tokens with it one at a time is slow, because generating token `t+1` requires a full forward pass, and you can't start on `t+2` until `t+1` is done. This sequential dependency is the fundamental bottleneck of autoregressive generation.
## The idea: let a cheap, fast model guess ahead, then verify in parallel

Speculative decoding uses a **small, fast "draft" model** to quickly propose several tokens in a row (cheap, since it's small). Then, the **expensive target model** checks all of these proposed tokens **at once**, in a single parallel forward pass (since scoring a fixed sequence of tokens for "how likely is each one" can be done in parallel — much cheaper than _generating_ them one at a time). If most of the draft's guesses turn out to be good, you've just gotten several tokens' worth of high-quality output for roughly the cost of one target-model forward pass.
## The hard part: how do you "verify" without introducing bias?

Here's the subtlety that makes this genuinely clever, not just a shortcut: you can't simply accept a draft token whenever the target model "likes it reasonably well" — that would silently bias your output toward whatever the _draft_ model tends to overproduce, corrupting the final distribution. The whole point is that **the final output must be statistically identical to what the target model alone would have produced** — speculative decoding needs to be a speed trick with **zero quality cost**, not an approximation.
## The acceptance rule — and why it's exactly this ratio

```
α_i = min(1, p_target(x_i) / p_draft(x_i))

```

Think about what this ratio is really measuring: **"did the draft model overestimate or underestimate how likely this token really is?"**

- If `p_target(x_i) ≥ p_draft(x_i)` — the target model likes this token **at least as much** as the draft did. There's no "overselling" happening, so you accept it **with certainty** (`α_i = 1`).
- If `p_target(x_i) &lt; p_draft(x_i)` — the draft model was **overconfident** about this specific token relative to what the target actually believes. You accept it only _proportionally_ — with probability equal to exactly how much the draft's confidence needs to be "corrected down" to match the target's true belief.

This is a form of **rejection sampling** — a classical statistics technique for "correcting" samples drawn from one distribution (draft) so they behave as if drawn from a different, target distribution — and it's mathematically guaranteed to work as long as you handle the rejected case correctly (next point).
## Why rejection needs a special "residual" distribution, not just a plain resample

If a token gets rejected, you can't just throw it away and sample fresh from `p_target` — that would **double-count** some probability mass, since the accept mechanism already "used up" some of the probability from tokens that matched. The correction needed is precisely:

```
p_residual(x) ∝ max(0, p_target(x) - p_draft(x))

```

**Intuition:** this represents _exactly_ the part of the target's belief that the draft model **failed to account for**. Wherever the draft already over-proposed a token, that excess doesn't need to be "given back" again — you only sample from the **leftover** mass that the draft's guess didn't already cover. `max(0, ...)` ensures you never get a negative "leftover" for tokens the draft already overrepresented. This residual, once normalized to sum to `1`, is a legitimate probability distribution to sample the replacement token from — and this exact construction is what makes the whole scheme mathematically prove out to reproduce `p_target` perfectly.
## Why a "bonus" token appears when everything is accepted

Here's a neat efficiency detail: to verify `k` draft tokens, the target model's single parallel forward pass naturally computes probability distributions for `k` positions — but also, as a byproduct, the target model has _already_ computed the distribution for **one more position beyond the last draft token** (since running the model on a sequence of length `k` naturally also tells you "what comes after position `k`"). If every single draft token got accepted, you get this `(k+1)`-th distribution **for free**, having already paid for the forward pass — so you sample one bonus token directly from it, extending your output by one more token at essentially no extra cost.## Steps to implement this

**Step 1: Loop through the draft tokens in order, one position at a time.** Unlike the MoE/attention problems where you processed everything in parallel, this algorithm is inherently **sequential** in its _decision-making_ — you must check position 0 before position 1, because the moment one rejection happens, everything after it is discarded (even if a later token _would_ have been accepted on its own).

**Step 2: At each position **`i`**, compute the acceptance probability.** `α_i = min(1, p_target(x_i) / p_draft(x_i))` — using the **specific proposed token **`x_i`'s probability under both distributions, not the full distributions themselves at this step.

**Step 3: Compare against the supplied uniform draw for this position.** Per the problem statement: accept if the draw is **strictly less than** `α_i`. This is the same inverse-CDF-style "supplied randomness" pattern from your earlier sampling problem — deterministic given the draw, but behaviorally equivalent to true random rejection sampling.

**Step 4: If accepted, record this token and continue to the next position.**

**Step 5: If rejected, stop processing further draft tokens immediately** — even if there are more tokens left to check. Then compute the residual distribution over the **full vocabulary** at this position: `max(0, p_target(x) - p_draft(x))` elementwise, then normalize by dividing by its own sum so it becomes a valid probability distribution.

**Step 6: Sample the replacement token from this residual distribution**, using that position's supplied uniform draw and the same inverse-CDF sampling technique you already built (cumulative sum, find first crossing point). This replacement is the **final** token for this sequence — no further draft tokens are considered after this.

**Step 7: If every draft token gets accepted (no rejection ever occurred), sample one bonus token.** This uses the target model's distribution at the position **immediately after** the last draft token — sampled fresh, with no draft distribution involved at all (since there was no draft proposal for this extra position).

**Step 8: Return the final sequence** — all accepted draft tokens, followed by either the residual replacement (if a rejection occurred) or the bonus token (if everything was accepted).
## A good sanity check once implemented

Try the edge case where `p_draft` and `p_target` are **identical** distributions at every position — in that case, `α_i = min(1, 1) = 1` for every token, meaning **every** draft token should always be accepted (never any rejection), since a perfect draft model needs no correction at all. This is a clean way to verify your acceptance logic before testing the trickier residual-sampling branch.

Want to try implementing this and share your attempt, the same way we've worked through the earlier problems?

-----------------------------------------------------------------------------------------------
Why rejection needs a special "residual" distribution, not just a plain resample
If a token gets rejected, you can't just throw it away and sample fresh from `p_target` — that would double-count some probability mass, since the accept mechanism already "used up" some of the probability from tokens that matched. The correction needed is precisely:

```
p_residual(x) ∝ max(0, p_target(x) - p_draft(x))
```

Let's use a simple, concrete numeric example — that'll make the "double-counting" problem click.
### **Setup: a tiny 2-token vocabulary**

Say there are only two possible tokens, "cat" and "dog."

```
p_draft:  cat = 0.7,  dog = 0.3
p_target: cat = 0.4,  dog = 0.6
```

The draft model really likes "cat" (`0.7`), but the target model actually prefers "dog" (`0.6`). The draft is "overselling" cat and "underselling" dog.
### **What we WANT, at the end of the day**

No matter what clever trick we use, the **final** token that comes out of this whole accept/reject process must behave _exactly_ as if we'd sampled directly from `p_target` — meaning, over many repeated trials, "cat" should be chosen `40%` of the time and "dog" `60%` of the time. That's the entire promise of speculative decoding: same final quality, just faster.
### **Walking through what actually happens, token by token**

**Case 1: draft proposes "cat" (happens 70% of the time, since that's **`p_draft(cat)`**)**

Acceptance probability: `α = min(1, p_target(cat)/p_draft(cat)) = min(1, 0.4/0.7) = 0.571`

So when draft proposes "cat": accept it `57.1%` of the time, reject it `42.9%` of the time.

**Case 2: draft proposes "dog" (happens 30% of the time)**

Acceptance probability: `α = min(1, p_target(dog)/p_draft(dog)) = min(1, 0.6/0.3) = min(1, 2.0) = 1.0`

So whenever draft proposes "dog," it's **always** accepted (since the target likes dog _even more_ than draft did — no correction needed downward).
### **Now the key question: what should happen on rejection?**

Rejection only ever happens in **Case 1** — when draft proposed "cat" and it got rejected. Let's count up, out of many trials, how "cat" actually gets accepted so far:

```
P(draft proposes cat AND it's accepted) = P(propose cat) × α_cat = 0.7 × 0.571 = 0.4
```

**Notice: this is exactly **`0.4`** — precisely **`p_target(cat)`**!** The accept mechanism, on its own, has _already_ delivered the correct amount of "cat" into the final output. If a rejection happens (draft said cat, but got rejected), and we then just resampled fresh from `p_target` — which still has `p_target(cat) = 0.4` sitting in it — we'd be **adding even more "cat"** on top of the `0.4` we already got exactly right. That's the "double-counting" — cat would end up overrepresented in the final output, breaking the guarantee that the output matches `p_target` exactly.
### **What the residual distribution actually contains**

```
p_residual(x) ∝ max(0, p_target(x) - p_draft(x))

residual(cat) = max(0, 0.4 - 0.7) = max(0, -0.3) = 0     &lt;- cat gets ZERO chance in the residual
residual(dog) = max(0, 0.6 - 0.3) = max(0, 0.3)  = 0.3    &lt;- dog gets all the leftover mass

normalize: residual(cat) = 0/0.3 = 0,  residual(dog) = 0.3/0.3 = 1.0
```

So the residual distribution says: **"if a rejection happens, always pick dog, never cat."** This makes complete sense — cat has _already_ been fully accounted for by the accept step (we got exactly `0.4` worth of it), so the residual correctly gives it **zero** further chance. Dog, on the other hand, was **underrepresented** by the draft (draft only gave it `0.3`, but target wants `0.6`) — so the residual is exactly the "missing `0.3`" that the accept/reject step alone couldn't deliver.
### **Confirming the full math works out to exactly **,`p_target`

```
P(final token = cat) = P(accepted directly) = 0.4                     [matches p_target(cat) = 0.4 exactly]

P(final token = dog) = P(draft=dog, accepted)  +  P(draft=cat, rejected) × P(residual picks dog)
                      = (0.3 × 1.0)            +  (0.7 × 0.429)        × 1.0
                      = 0.3                    +  0.3
                      = 0.6                                            [matches p_target(dog) = 0.6 exactly]
```

Both come out **exactly right** — `0.4` for cat, `0.6` for dog — precisely matching `p_target`. This is why the residual has to be `max(0, p_target - p_draft)`, normalized: it's precisely the _leftover_ probability mass that the accept/reject mechanism didn't already deliver on its own, so adding it back in (only on rejection) completes the picture perfectly without ever over- or under-counting any token.
### **The one-line intuition**

> The accept step already gives each token exactly `min(p_target, p_draft)` worth of representation "for free." The residual distribution exists purely to make up the **difference** — `p_target - p_draft`, wherever target wanted more than draft gave — and it deliberately excludes any token where draft already gave _enough_ (or too much), since giving more there would overshoot the target.