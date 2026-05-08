# Method

Full specification of the two-layer Critical Slowing Down (CSD) pipeline.
This document complements the [README](README.md); for context and
results, start there.

---

## 1. From utterance to continuous affect

Each utterance is projected to a triplet $(V, A, D) \in [-1, +1]^3$ via the
**NRC VAD Lexicon v2.1**, using a multi-word-expression-first greedy match:

1. Lower-case and keep alphabetic tokens.
2. At each position $i$, try the bigram $w_i\ w_{i+1}$ first; if it hits the
   lexicon, record its (V, A, D) and advance two tokens.
3. Otherwise try the unigram $w_i$; if it hits, record and advance one token.
4. Otherwise skip.

The utterance VAD is the mean across matched lexicon entries. Utterances
with zero coverage receive same-speaker, same-episode mean imputation.

The single axis of analysis is the **Signed Emotional Potential**:

$$\mathrm{SEP}_t = V_t + 0.5 \cdot D_t$$

Valence is the primary signed-affect signal; dominance contributes secondary
directional weight. Arousal is excluded from SEP because it is not
directional — high arousal can be excitement or panic, and folding it in
collapses two opposite signs.

For each character $c$ the equilibrium $\mu_c$ and spread $\sigma_c$ are
computed across that character's lexicon-matched utterances.

---

## 2. Two-layer detector

A 5-utterance rolling window (respecting episode boundaries) yields three
derived signals at every step:

- **Smoothed deviation** $\bar{\Delta}_t = \mathrm{mean}\bigl(\mathrm{SEP}_{t-k+1:t}\bigr) - \mu_c$
- **Lag-1 autocorrelation** $r_1(t)$
- **Variance** $\mathrm{Var}(t)$

### Layer 1 — sustained negative deviation

$$\bar{\Delta}_t < -0.5 \cdot \sigma_c$$

The state-variable condition: the speaker is meaningfully below their own
emotional baseline.

### Layer 2 — joint CSD elevation

$$r_1(t) > Q_{75}\bigl(r_1\bigr) \ \wedge\ \mathrm{Var}(t) > Q_{75}\bigl(\mathrm{Var}\bigr)$$

Both signals exceed the speaker's own 75th-percentile simultaneously. This
is the textbook CSD early-warning signature: recovery from perturbation has
slowed (high $r_1$) and the system is fluctuating between basins (high variance).

A one-step lead-lag tolerance allows Layer 2 to fire just before Layer 1,
mirroring the physical-systems property that CSD signals *precede* the
state-variable nadir.

### Segment extraction

Contiguous runs of $\geq 3$ same-episode utterances satisfying both layers
form a candidate **tipping segment**. Segments never cross episode
boundaries.

---

## 3. Validation

### Post-hoc recovery time τ

For each detected segment, the recovery time $\tau$ is the number of
utterances after the segment's nadir before SEP returns within
$0.3 \cdot \sigma_c$ of $\mu_c$. The diagnostic is the τ-ratio:

$$\tau\text{-ratio} = \frac{\tau_{\text{inside detected segment}}}{\tau_{\text{baseline}}}$$

A τ-ratio > 1.5× indicates genuinely impaired return to baseline — the
dynamical-systems definition of having entered a different basin.

### Manual validation codebook

A segment is judged a **true positive** if all five criteria hold:

1. **Sustained negative affect** across at least 3 consecutive utterances by the target (Bowlby's protest–despair temporal arc).
2. **Failure of soothing** — comforting attempts by another character are rejected, dismissed, or fail to lift the trajectory.
3. **Beck cognitive triad** — explicit verbal endorsement of negative views of self, world, or future.
4. **Rumination markers** — repeated re-expression of the same distress content (Nolen-Hoeksema et al., 2008).
5. **At least one of:** elevated first-person-singular pronoun use (Rude et al., 2004) or absolutist vocabulary (Al-Mosaiwi & Johnstone, 2018).

Manual review by the author, with a blind second-rater pilot on a 6-segment
subset (κ = 0.975).

---

## 4. Robustness checks

Both the SEP weight $\alpha$ in $\mathrm{SEP} = V + \alpha \cdot D$ and the
rolling window size $k$ were swept:

| Parameter | Range | Effect |
|---|---|---|
| $\alpha$ (dominance weight) | 0.3, 0.4, 0.5, 0.6, 0.7 | Total segment count stays in 20–22; 60–74% pairwise overlap with $\alpha = 0.5$ |
| $k$ (rolling window) | 3, 5, 7, 10 | $k=5$ gives the highest mean per-character segment count and Sad/Mad enrichment |

The qualitative findings are not artefacts of the specific parameter choice.

---

## 5. Failure-mode taxonomy

The 14 false positives in the *Friends* validation partition into six modes:

| Failure mode | Count | Source | Tunable? |
|---|:---:|---|---|
| Cross-scene contamination | 4 | Rolling window straddles two narratively distinct scenes | yes (window-side) |
| **Emotion attribution error** | **4** | Target speaker is responding to another character's emotional event | **no — structural** |
| Defensive humour | 2 | Lexicon × register: ironic / sarcastic distress flattens to neutral VAD | partial (lexicon-side) |
| Lexicon error | 2 | Word-level VAD misrepresents contextual meaning ("too bad" as polite sympathy) | partial (lexicon-side) |
| Normal trajectory | 2 | Threshold sensitivity at the 0.5σ boundary | yes (threshold-side) |
| Statistical noise | 2 | Joint Layer-1 + Layer-2 condition met without a recognisable narrative event | partial |

**Emotion attribution** is the focus of the case study and the AI-alignment
extension: it is a structural property of multi-party dialogue, not a
parameter problem.

---

## References

- van de Leemput, I. A. et al. (2014). *Critical slowing down as early warning for the onset and termination of depression.* PNAS 111(1).
- Mohammad, S. M. (2018). *Obtaining Reliable Human Ratings of Valence, Arousal, and Dominance for 20,000 English Words.* Proc. ACL.
- Zahiri, S. and Choi, J. D. (2018). *Emotion Detection on TV Show Transcripts with Sequence-Based Convolutional Neural Networks.* Proc. AAAI Workshops.
- Beck, A. T. (1967). *Depression: Clinical, Experimental, and Theoretical Aspects.*
- Nolen-Hoeksema, S., Wisco, B. E., and Lyubomirsky, S. (2008). *Rethinking rumination.* Perspectives on Psychological Science 3(5).
- Rude, S., Gortner, E.-M., and Pennebaker, J. (2004). *Language use of depressed and depression-vulnerable college students.* Cognition and Emotion 18(8).
- Al-Mosaiwi, M. and Johnstone, T. (2018). *In an absolute state: Elevated use of absolutist words is a marker specific to anxiety, depression, and suicidal ideation.* Clinical Psychological Science 6(4).
- Bowlby, J. (1980). *Attachment and Loss, Vol. III: Loss, Sadness and Depression.*
