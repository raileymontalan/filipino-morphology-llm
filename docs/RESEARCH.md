# Research Overview

## Research Question

**Can tokenization that preserves morpheme boundaries improve LLM performance on Filipino morphological understanding tasks?**

---

## Hypothesis

Standard BPE tokenization (GPT-2, Gemma, LLaMA) systematically destroys morpheme boundaries in agglutinative languages like Filipino. By applying **morpheme-aware tokenization** during continued pretraining, we hypothesize:

1. Models will better preserve morpheme boundaries
2. Performance on morphological tasks will improve
3. Models will develop stronger Filipino linguistic representations

---

## Experimental Design

### Three Tokenization Approaches

| Approach | Method | Status |
|----------|--------|--------|
| **Baseline** | Standard BPE (no modification) | ✅ Implemented |
| **Stochastok** | Stochastic token expansion (~10%) | ✅ Implemented |
| **Patok** | Affix-aware expand-contract (30%+30%) | 🚧 In development |

---

## 1. Baseline (Vanilla BPE)

**Method:** Use pretrained tokenizer as-is

**Implementation:** Default NeMo behavior in `training/nemo/run_cpt.py`

**Characteristics:**
- Fast preprocessing
- No morphological awareness
- Standard baseline for comparison

**Example:**
```
Text:  "Kumain ako ng pagkain"
Tokens: [K][um][ain][ ako][ ng][ pag][kain]
```

Morpheme boundaries: Not preserved

---

## 2. Stochastok (Token Expansion)

**Method:** Randomly expand ~10% of tokens by splitting them into sub-tokens

**Implementation:** `src/tokenization/stochastok_processor.py`

**How it works:**
1. Tokenize text normally
2. Build expansion dictionary from tokenizer's merge vocabulary
3. Randomly select ~10% of tokens
4. Split selected tokens into sub-tokens
5. Result: Longer sequences with finer granularity

**Example:**
```
Original: [K][um][ain]
Expanded: [K][u][m][ain]  (split "um" → "u" + "m")
```

**Expected benefits:**
- Forces model to learn sub-token structure
- Improves robustness to tokenization variations
- Better character-level understanding

**Trade-offs:**
- 10-15% longer sequences
- Slower training (~10% overhead)
- 2x slower preprocessing

---

## 3. Patok (Affix-Aware) - In Development

**Method:** Expand + contract with preference for Filipino morpheme boundaries

**Implementation:** `src/tokenization/patok_processor.py`

**How it works:**
1. Load Filipino affixes from `data/affixes/filipino_affixes.txt`
2. Expand ~30% of tokens (split into sub-tokens)
3. Contract ~30% of adjacent tokens (merge pairs)
4. Prefer contractions that form valid affixes (70% probability)
5. Result: Sequences aligned with morpheme boundaries

**Example:**
```
Original:    [K][um][ain]
Expand:      [K][u][m][a][in]  (expand more aggressively)
Contract:    [K][um][ain]      (but prefer forming "um" affix)
Result:      Morpheme-aware tokenization
```

**Key Innovation: Affix Preference**
```python
# When contracting adjacent tokens
if merged_token in affix_vocabulary:
    # 70% chance to choose this contraction
    prefer_this_merge()
else:
    # 30% chance to choose random merge
    random_merge()
```

**Expected benefits:**
- Tokens align with morpheme boundaries
- Model sees morphologically meaningful units
- Improved affix understanding
- Better performance on morphological tasks

**Trade-offs:**
- More complex preprocessing
- Requires affix dictionary
- Similar sequence lengths to baseline (expand + contract balance)

---

## Evaluation Framework

### PACUTE Benchmark (11,225 tasks)

Tests morphological understanding across 4 dimensions:

1. **Affixation** (280 tasks): Identify and apply Filipino affixes
2. **Composition** (3,905 tasks): Character counting, diacritics, word formation
3. **Manipulation** (5,120 tasks): Character operations (insert, delete, swap)
4. **Syllabification** (1,280 tasks): Syllable counting, stress, reduplication

### Hierarchical Benchmark (1,798 tasks)

**Purpose:** Diagnose where models fail in linguistic hierarchy

**6 Compositional Levels:**

```
Level 0: Character Recognition
    ↓
Level 1: Character Manipulation (requires Level 0)
    ↓
Level 2: Morpheme Decomposition (requires Level 0)
    ↓
Level 3: Morpheme Manipulation (requires Level 1 + Level 2)
    ↓
Level 4: Morpheme Composition (requires Level 2)
    ↓
Level 5: Complex Reasoning (requires Level 2-4)
```

#### Diagnostic Cascade

If a model fails at Level N, we expect failures at dependent levels.

**Example Analysis:**
- ✅ Level 0 (95%): Model can recognize characters
- ✅ Level 1 (75%): Model can manipulate strings
- ❌ Level 2 (40%): **Bottleneck: morpheme decomposition**
- ❌ Level 3 (25%): Expected failure (needs Level 1 + Level 2)
- ❌ Level 4 (20%): Expected failure (needs Level 2)
- ❌ Level 5 (15%): Expected failure (needs Level 2-4)

**Diagnosis:** Tokenization doesn't align with morpheme boundaries

**Solution:** Use Stochastok or Patok to improve Level 2

#### Level Descriptions

**Level 0: Character Recognition**
- Test if model has access to individual characters
- Tasks: "What is the 3rd character in 'kumain'?" → 'm'
- Failure mode: Tokenization too coarse-grained

**Level 1: Character Manipulation**
- Test if model can perform character operations
- Tasks: "Delete the 3rd character from 'kumain'" → "kuain"
- Requirements: Level 0
- Failure mode: Can see but not manipulate characters

**Level 2: Morpheme Decomposition** ⚠️ **Critical bottleneck**
- Test if model understands morphological boundaries
- Tasks: "What is the infix in 'kumain'?" → "um"
- Failure mode: Tokenization ignores morphology
- **This is where affix-aware tokenization helps most**

**Level 3: Morpheme Manipulation**
- Test if model can transform morphological units
- Tasks: "Change 'um' to 'mag' in 'kumain'" → "magkain"
- Requirements: Level 1 + Level 2
- Failure mode: Can't identify and manipulate morphemes

**Level 4: Morpheme Composition**
- Test if model can combine morphemes correctly
- Tasks: "Combine 'ka-' + 'alis' + '-an'" → "kaalisan"
- Requirements: Level 2
- Failure mode: Can't compose morphemes properly

**Level 5: Complex Morphological Reasoning**
- Test advanced morphological understanding
- Tasks: "Apply actor focus to 'kain'" → "kumain"
- Requirements: Level 2-4
- Failure mode: Lacks deep morphological knowledge

---

## Expected Results

### Baseline (Vanilla BPE)
- PACUTE Affixation: 40-50% (poor morphological understanding)
- Hierarchical Level 2: 30-40% (morpheme boundary issues)
- Hierarchical Levels 3-5: <30% (cascading failures)

### Stochastok (+10-15% improvement)
- PACUTE Affixation: 50-65% (+10-15%)
- Hierarchical Level 2: 45-55% (+15%)
- Better character-level understanding
- Reduced cascading failures

### Patok (+20-30% improvement - expected)
- PACUTE Affixation: 60-70% (+20-30%)
- Hierarchical Level 2: 55-70% (+25%)
- Strong morpheme boundary alignment
- Significant reduction in cascading failures

---

## Implementation Details

### Stochastok Processor

**File:** `src/tokenization/stochastok_processor.py`

**Key methods:**
```python
class StochastokProcessor:
    def __init__(self, tokenizer, expand_prop=0.1):
        self.tokenizer = tokenizer
        self.expand_prop = expand_prop
        self.set_expansions()  # Build expansion dictionary
    
    def expand(self, token_ids, expand_prop=0.1):
        # Randomly expand tokens
        num_to_expand = int(len(token_ids) * expand_prop)
        for _ in range(num_to_expand):
            # Select random token
            # Split if expandable
            # Replace in sequence
        return expanded_ids
    
    def build_expansions(self):
        # Analyze tokenizer's merge vocabulary
        # Build dictionary: token_id → [possible splits]
        return expansions
```

### Patok Processor

**File:** `src/tokenization/patok_processor.py`

**Key methods:**
```python
class PatokProcessor:
    def __init__(self, tokenizer, affixes_file,
                 expand_prop=0.3, contract_prop=0.3, 
                 affix_preference=0.7):
        self.tokenizer = tokenizer
        self.expand_prop = expand_prop
        self.contract_prop = contract_prop
        self.affix_preference = affix_preference
        self.load_affixes(affixes_file)
        self.build_expansions()
        self.build_contractions()
    
    def load_affixes(self, affixes_file):
        # Load Filipino affixes
        # Convert to token IDs
        # Store in self.affix_token_ids
    
    def affix_aware_expand_contract(self, token_ids, num_iterations=3):
        for _ in range(num_iterations):
            # Expand phase: split tokens (avoid affixes)
            token_ids = self._selective_expand(token_ids)
            # Contract phase: merge pairs (prefer affixes)
            token_ids = self._affix_preferring_contract(token_ids)
        return token_ids
    
    def _selective_expand(self, token_ids):
        # Identify expandable tokens
        # Prefer non-affix tokens for expansion
        # Fallback to affix tokens if needed
        return expanded_ids
    
    def _affix_preferring_contract(self, token_ids):
        # Find all contractable pairs
        # Categorize: affix-forming vs regular
        # Use affix_preference probability
        # Prefer affix-forming contractions
        return contracted_ids
```

**Validity guarantees:**
- All tokens remain valid (in vocabulary)
- Sequences are always decodable
- No invalid tokens created

---

## Research Pipeline

```
1. Data Collection
   ↓ SEA-PILE v2 (7.4GB Filipino text)
   
2. Preprocessing
   ↓ Three parallel tracks:
   ├─ Baseline (vanilla BPE)
   ├─ Stochastok (expand 10%)
   └─ Patok (expand 30% + contract 30% with affix pref)
   
3. Training (NeMo CPT)
   ↓ Gemma 3 1B (1B parameters)
   ↓ 10K steps per model
   
4. Evaluation
   ↓ PACUTE + Hierarchical benchmarks
   ↓ 13,023 total tasks
   
5. Analysis
   ↓ Compare performance across tokenization methods
   ↓ Identify bottlenecks via hierarchical levels
   ↓ Measure improvement from affix-aware tokenization
   
6. Publication
   └─ Document findings and release benchmarks
```

---

## Key Insights

### Why Filipino?
- **Agglutinative morphology**: Rich affix system
- **Clear morpheme boundaries**: Affixes are well-defined
- **BPE misalignment**: Standard tokenizers ignore morphology
- **Measurable impact**: Morphological tasks show clear differences

### Why Hierarchical Evaluation?
- **Precise diagnosis**: Identify exact failure points
- **Compositional structure**: Test dependencies
- **Actionable insights**: Know what to fix
- **Research contribution**: Novel evaluation framework

### Why Affix-Aware Tokenization?
- **Alignment with linguistics**: Respects morpheme boundaries
- **Measurable benefit**: +20-30% on morphological tasks (expected)
- **Language-specific**: Leverages Filipino linguistic knowledge
- **Generalizable**: Method applicable to other agglutinative languages

---

## Future Work

1. **Extend to other agglutinative languages:**
   - Tagalog variations
   - Indonesian/Malay
   - Turkish
   - Finnish

2. **Improve Patok algorithm:**
   - More sophisticated affix detection
   - Context-aware expansion/contraction
   - Multi-affix handling

3. **Larger-scale experiments:**
   - Bigger models (7B, 13B parameters)
   - More training data
   - Longer training (100K+ steps)

4. **Additional evaluation:**
   - Downstream tasks (NLU, QA)
   - Cross-lingual transfer
   - Real-world applications

---

For setup instructions, see `SETUP.md`.  
For training procedures, see `docs/TRAINING.md`.  
For evaluation procedures, see `docs/EVALUATION.md`.
