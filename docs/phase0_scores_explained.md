# What the LLM-Check scores compute (Phase 0 whiteboard notes)

All five scores come from **one forward pass with teacher forcing**: we feed
`[chat template + prompt + passages + answer]` into the reader model and read
its internals while it *reads* the answer. Nothing is generated. Every score is
computed only on the answer tokens `[i1, i2)` (`response_span` in
`rag_halluc/models.py`).

Notation: `T = i2 - i1` answer tokens, vocabulary size `V`, `H` attention
heads, hidden size `d`. `p_t` is the model's predicted distribution for the
token at position `t` (it comes from the logits at position `t-1`).

Code references are to the original, unmodified `common_utils.py`.

---

## 1. Perplexity (`perplexity`)

```
log p_t(x_t) = log softmax(logits[t-1])[x_t]          for t in [i1, i2)
perplexity   = exp( - mean_t log p_t(x_t) )
```

* The code indexes `logits[torch.arange(i1, i2) - 1, tok_ins[0, i1:i2]]`: the
  `-1` is because the logits at position `t-1` predict token `t`.
* **Intuition:** how surprised the reader is by the answer, on average. A
  claim the passages do not support should be less predictable once the reader
  has seen the passages, so higher perplexity is expected for hallucinations.
* Range: 1 (perfectly predicted) upwards.

## 2. Logit entropy (`logit_entropy`, top-k = 50)

```
q_t = softmax( top-50 logits at position t )           (renormalised over the 50)
score = mean over t and over the 50 entries of  -q_t log q_t
      = (1/50) * mean_t H(q_t)
```

* **Intuition:** how *unsure* the reader is about what comes next, regardless
  of which token actually came. High entropy = the model sees many plausible
  continuations, typical when it has no evidence to copy from.
* Difference from perplexity: perplexity looks at the probability of the
  *actual* token; entropy looks at the *spread* of the whole distribution.
* Two quirks worth knowing (they do not change AUROC much):
  1. it uses `logits[i1:i2]`, i.e. one position later than perplexity
     (it skips the prediction of the first answer token and includes the
     prediction after the last one);
  2. `.mean()` averages over the 50 entries too, so the value is the entropy
     divided by 50. A constant factor never changes AUROC.

## 3. Window entropy (`window_logit_entropy`, w = 1)

```
h_t   = mean over the full vocabulary of -p_t log p_t     (= H(p_t) / V)
score = max over windows of w tokens of the mean h_t      (w = 1: max_t h_t)
```

* **Intuition:** a hallucination is often one or two invented tokens (a
  wrong date, a name) inside an otherwise fluent answer. Averaging hides
  them; taking the **maximum** over short windows finds the single most
  uncertain spot.

## 4. Attention score (`get_attn_eig_prod`) -- the "eigen" score

For one layer, each head `h` gives an attention matrix `A_h` restricted to the
answer tokens (a `T x T` block):

```
score(layer) = sum over heads h of  mean_t log A_h[t, t]
```

Why this is a log-determinant (the key whiteboard step):

1. Causal attention: token `t` may only attend to tokens `<= t`, so every
   entry above the diagonal is 0. `A_h` is **lower-triangular**.
2. The eigenvalues of a triangular matrix are its diagonal entries, and the
   determinant is their product: `det A_h = prod_t A_h[t, t]`.
3. So `log det A_h = sum_t log A_h[t, t]`. LLM-Check uses the mean instead of
   the sum (`log det / T`) so that long and short answers are comparable.
   No eigen-decomposition is needed: it costs `O(T)` per head.

* `A_h[t, t]` is how much token `t` attends **to itself**. The score is the
  average log self-attention, summed over heads.
* **Intuition (from the paper):** truthful and hallucinated answers produce
  measurably different attention "volumes"; the score is a cheap summary of
  that geometry. It does **not** look at the passages at all -- which is the
  gap our Phase 2 "context-aware" variant fills (attention from answer tokens
  to the passage block, and lookback ratio).
* `tests/test_phase0.py::test_attention_logdet_equals_sum_of_log_diagonal`
  checks steps 1 to 3 numerically on a real attention map.
* Implementation note: in fp16 a tiny self-attention weight can round to 0 and
  `log 0 = -inf`, so our script floors weights at `1e-8` before calling the
  function (the original ran in bf16, which has a wider exponent range).

## 5. Hidden-state score (`get_svd_eval` -> `centered_svd_val`)

For one layer take the answer's hidden states, a `T x d` matrix, transpose to
`Z` (`d x T`), and:

```
J      = I_d - (1/d) 1 1^T            centring matrix
Sigma  = Z^T J Z + alpha I_T           T x T Gram matrix of centred token vectors, alpha = 0.001
score  = mean of log singular values of Sigma  = (1/T) log det Sigma
```

* `Sigma[s, t]` is the dot product between the (centred) representations of
  answer tokens `s` and `t`. Its log-determinant measures the **volume** the
  token representations span: diverse, spread-out representations give a large
  volume; collapsed, similar ones give a small volume.
* **Intuition:** the same idea as INSIDE / EigenScore (Chen et al., 2024), but
  on the tokens of one answer instead of across several sampled answers.
* `alpha I` keeps `Sigma` invertible (it is rank-deficient when `T > d` or
  tokens are near-identical). Because `Sigma` is symmetric positive-definite,
  singular values = eigenvalues, so the mean log singular value is exactly
  `log det / T`.
* Quirk: `J` is `d x d`, so the centring removes each token's mean over the
  hidden dimensions, not the mean over tokens.
* It needs an SVD of a `T x T` matrix per layer, which is why it is the
  slowest of the five (and is optional in Phase 0: add `hidden` to `scores`).

---

## How we turn scores into numbers (`rag_halluc/eval/metrics.py`)

* Label 1 = the answer has at least one hallucinated span in RAGTruth.
* **AUROC**: the probability that a random hallucinated answer gets a higher
  score than a random correct one. 0.5 = chance, 1.0 = perfect.
* The scores are unsupervised, so their direction is unknown in advance. We
  report the raw AUROC (below 0.5 = the score works upside down), the
  direction, and `auroc_best = max(auroc, 1 - auroc)`.
* `bal_acc` = best balanced accuracy over thresholds; `tpr_at_5fpr` = recall
  when at most 5% of correct answers are wrongly flagged. Both follow the
  definitions in LLM-Check's `get_roc_scores`.

## Questions a panel may ask about Phase 0

* *Why one forward pass and not generation?* The answer already exists; we
  only need the model's view of it. Teacher forcing gives all logits,
  attentions and hidden states in one pass, which is what makes LLM-Check
  cheap (and our cascade fast).
* *Why `attn_implementation="eager"`?* Fast attention kernels (SDPA / Flash)
  never materialise the attention matrix, so they cannot return it.
* *Why can a model that did not write the answer detect its hallucinations?*
  It reads the passages first; tokens that are not grounded in them are
  less predictable (perplexity, entropy) and change the attention and hidden
  geometry. Phase 2 makes the passage dependence explicit.
* *Why are Phase 0 numbers lower than the paper's?* Smaller readers, a
  different reader from the generator, and only 200 answers (AUROC on 200
  samples has a 95% interval of roughly +/- 0.08).
