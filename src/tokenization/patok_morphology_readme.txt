To quickly initialize a MorephologyAwarePatokProcessor from this folder
(assuming you'll be working with Gemma):

```
import json
import numpy as np
import random
from tqdm import tqdm
from patok_morphology import MorphologyAwarePatokProcessor

expansions = './expansions/expansions_gemma.json'

processor = MorphologyAwarePatokProcessor(
                                            tokenizer,
                                            prefix_file='./affixes/prefix.txt',
                                            infix_file='./affixes/infix.txt',
                                            suffix_file='./affixes/suffix.txt',
                                            expansions_file = expansions
                                         )
```

Other expansions: expansions_llama.json, expansions_qwen.json,
                  expansions_gpt2.json, expansion_gpt_oss.json

Other arguments to play around with:
- num_toks_to_cont (default: [2,3,4]): list of possibilities for number of tokens to merge
- contract_prob (default: [0.35,0.35,0.3]): probability weights of choosing number of tokens in num_toks_to_cont
- affix_awareness (default: 0.95): probability of skipping contraction if token is affix;
                                   probability affix will be split off from contracted token
- affix_awareness_if_overlap (default: 0.75): affix awareness if multiple affixes at same position
- expand_prop (default: 0.1): Default proportion of tokens to expand
- contract_prop (default 0.9): Default proportion of tokens to contract

==========================================================================================================================

To expand a vanilla-tokenized string using MorphologyAwarePatokProcessor:

from transformers import AutoTokenizer

model = "google/gemma-3-270m"
tokenizer = AutoTokenizer.from_pretrained(model)

text = 'sample_string'
token_ids = tokenizer.encode(text)
expanded_ids = processor.contract_expand(token_ids)
