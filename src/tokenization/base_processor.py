"""
Base class for tokenizer processors with common utilities.
"""

import os


class TokenizerProcessor:
    """
    Base class for tokenizer processors that provides common utilities.
    
    Attributes:
        tokenizer: The HuggingFace tokenizer instance
        tokenizer_name: Sanitized tokenizer name for use in filenames
    """
    
    def __init__(self, tokenizer):
        """
        Initialize the base processor.
        
        Args:
            tokenizer: HuggingFace tokenizer instance
        """
        self.tokenizer = tokenizer
        self.tokenizer_name = self._get_tokenizer_name()
    
    def _get_tokenizer_name(self):
        """
        Get a safe filename from tokenizer name or config.
        
        Returns:
            str: Sanitized tokenizer name with / and \ replaced by -
        """
        # Try to get name from tokenizer attributes
        if hasattr(self.tokenizer, 'name_or_path'):
            name = self.tokenizer.name_or_path
        elif hasattr(self.tokenizer, 'model_name'):
            name = self.tokenizer.model_name
        else:
            name = self.tokenizer.__class__.__name__
        
        # Sanitize for filename (replace / with -)
        return name.replace('/', '-').replace('\\', '-')
    
    def get_mergeable_ranks(self):
        """
        Get mergeable ranks (token bytes -> token ID mapping) for the tokenizer.
        
        This function provides compatibility across different tokenizer implementations:
        - For tiktoken-based tokenizers (e.g., GPT-4), uses the `_mergeable_ranks` attribute
        - For other tokenizers (e.g., GemmaTokenizerFast), constructs the mapping from vocab
        
        Returns:
            dict: Mapping from token bytes to token IDs {bytes: int}
            
        Note:
            Some tokenizers like Gemma use byte-fallback BPE where not all tokens
            can be cleanly represented as UTF-8 strings. This function handles such cases.
        """
        # Check if tokenizer has tiktoken's _mergeable_ranks attribute
        if hasattr(self.tokenizer, '_mergeable_ranks'):
            return self.tokenizer._mergeable_ranks
        
        # Fall back to building from vocab for other tokenizers
        print("Building mergeable_ranks from tokenizer vocab (tokenizer doesn't have _mergeable_ranks)...")
        vocab = self.tokenizer.get_vocab()
        mergeable_ranks = {}
        
        for token_str, token_id in vocab.items():
            try:
                # Try to encode the token string to bytes
                token_bytes = token_str.encode('utf-8')
                mergeable_ranks[token_bytes] = token_id
            except (UnicodeDecodeError, UnicodeEncodeError, AttributeError):
                # Skip tokens that can't be encoded (e.g., special tokens, byte-fallback tokens)
                continue
        
        print(f"Built mergeable_ranks with {len(mergeable_ranks)} entries from vocab of size {len(vocab)}")
        return mergeable_ranks
    
    def get_cache_path(self, subdirectory, filename):
        """
        Get the full path for a cache file in the project root data directory.
        
        Args:
            subdirectory: Subdirectory under data/ (e.g., "tokenizer_expansions")
            filename: Name of the cache file
            
        Returns:
            str: Absolute path to the cache file
        """
        # Get project root (two levels up from this file: src/tokenization -> src -> root)
        current_file_path = os.path.abspath(__file__)
        src_tokenization_dir = os.path.dirname(current_file_path)
        src_dir = os.path.dirname(src_tokenization_dir)
        project_root = os.path.dirname(src_dir)
        
        # Build path to data directory in project root
        cache_path = os.path.join(project_root, "data", subdirectory, filename)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
        return cache_path
