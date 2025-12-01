#!/usr/bin/env python3
"""
Generate syllabified dictionaries for Indonesian and Malaysian languages.

This script processes dictionary data for Indonesian (from KBBI) and Malaysian (from DBP)
languages, extracting or generating syllabification for each word. The output is saved
as JSONL files containing word, syllable list, and metadata.

Usage:
    python scripts/generate_dictionaries.py
    python scripts/generate_dictionaries.py --id-input data/corpora/kbbi_v.csv
    python scripts/generate_dictionaries.py --ms-input data/corpora/dbp-v6.json
    python scripts/generate_dictionaries.py --output-dir data/dictionaries
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import re
from pathlib import Path
import pandas as pd

from src.corpus_generation.syllabifiers import MalaysianSyllabifier


def prepare_id(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare Indonesian dictionary data with syllabification.
    
    Extracts syllabification from the 'nama' column which contains
    dot-separated syllables (e.g., 'a.cang').
    
    Args:
        dataframe: Input DataFrame with 'key' (word) and 'nama' (syllables) columns
        
    Returns:
        Processed DataFrame with word, syllable_list, last_syllable, and last_syllable_index
    """
    df = dataframe.copy()
    df = df.rename(columns={"key": "word", "nama": "syllable_list"})
    df = df[["word", "syllable_list"]]
    df['word'] = df['word'].str.lower()
    df['syllable_list'] = df['syllable_list'].str.lower()

    # Filter valid entries
    df = df.dropna(subset=['syllable_list'])
    df = df.drop_duplicates(subset=['word'])
    df = df.drop_duplicates(subset=['syllable_list'])
    df = df[df['syllable_list'].str.contains('.', regex=False)]
    df = df[df['word'].str.contains('^[a-zA-Z]+$', regex=True)]
  
    # Parse syllables
    df['syllable_list'] = df['syllable_list'].apply(lambda x: x.split('.'))
    df['last_syllable'] = df['syllable_list'].apply(lambda x: x[-1])
    df['last_syllable_index'] = df.apply(lambda row: len(row['syllable_list']) - 1, axis=1)
    
    return df.reset_index(drop=True)


def prepare_ms(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare Malaysian dictionary data with syllabification.
    
    Extracts syllabification from the 'definisi' column if present in square brackets,
    otherwise uses the MalaysianSyllabifier to generate syllables.
    
    Args:
        dataframe: Input DataFrame with 'word' and 'definisi' (definition) columns
        
    Returns:
        Processed DataFrame with word, syllable_list, last_syllable, and last_syllable_index
    """
    syllabifier = MalaysianSyllabifier()
    
    df = dataframe.copy()
    df = df.rename(columns={"definisi": "definition"})
    df = df[["word", "definition"]]
    df['word'] = df['word'].str.lower()
    
    # Remove rows with missing definitions
    df = df.dropna(subset=['definition'])
    
    # Function to extract syllables from definition or use syllabifier
    def get_syllables(row):
        word = row['word']
        definition = row['definition']
        
        # Only process words that contain only letters
        if not re.match(r'^[a-zA-Z]+$', word):
            return None
        
        # Try to extract syllabification from square brackets
        if isinstance(definition, str):
            match = re.search(r'\[([^\]]+)\]', definition)
            if match:
                syllable_text = match.group(1)
                # Check if it contains dots (syllable separators)
                if '.' in syllable_text:
                    syllables = [s.strip() for s in syllable_text.split('.')]
                    # Validate that syllables when joined match the word
                    if ''.join(syllables).lower() == word.lower():
                        return syllables
        
        # Handle list type definitions (extract first element)
        if isinstance(definition, list) and len(definition) > 0:
            definition_str = str(definition[0])
            match = re.search(r'\[([^\]]+)\]', definition_str)
            if match:
                syllable_text = match.group(1)
                if '.' in syllable_text:
                    syllables = [s.strip() for s in syllable_text.split('.')]
                    if ''.join(syllables).lower() == word.lower():
                        return syllables
        
        # Fallback: use MalaysianSyllabifier
        try:
            syllables = syllabifier.syllabify(word)
            return syllables if syllables else None
        except:
            return None
    
    # Apply syllabification
    df['syllable_list'] = df.apply(get_syllables, axis=1)
    
    # Remove rows where syllabification failed
    df = df.dropna(subset=['syllable_list'])
    
    # Drop duplicates
    df = df.drop_duplicates(subset=['word'])
    
    # Add last syllable information
    df['last_syllable'] = df['syllable_list'].apply(lambda x: x[-1] if x else None)
    df['last_syllable_index'] = df['syllable_list'].apply(lambda x: len(x) - 1 if x else None)
    
    # Select final columns
    df = df[['word', 'syllable_list', 'last_syllable', 'last_syllable_index']]
    
    return df.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(
        description="Generate syllabified dictionaries for Indonesian and Malaysian"
    )
    parser.add_argument(
        "--id-input",
        type=str,
        default="data/corpora/kbbi_v.csv",
        help="Path to Indonesian (KBBI) dictionary CSV file"
    )
    parser.add_argument(
        "--ms-input",
        type=str,
        default="data/corpora/dbp-v6.json",
        help="Path to Malaysian (DBP) dictionary JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/dictionaries",
        help="Output directory for generated dictionary files"
    )
    parser.add_argument(
        "--skip-id",
        action="store_true",
        help="Skip Indonesian dictionary generation"
    )
    parser.add_argument(
        "--skip-ms",
        action="store_true",
        help="Skip Malaysian dictionary generation"
    )

    args = parser.parse_args()

    # Setup paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Dictionary Generation")
    print("=" * 70)
    print()

    # Process Indonesian dictionary
    if not args.skip_id:
        id_input_path = Path(args.id_input)
        print(f"Processing Indonesian dictionary from {id_input_path}...")
        
        if not id_input_path.exists():
            print(f"  ✗ Error: Input file not found: {id_input_path}")
            print(f"    Please ensure the KBBI dictionary CSV file exists.")
            return 1
        
        try:
            # Load Indonesian data
            id_words = pd.read_csv(id_input_path)
            print(f"  Loaded {len(id_words):,} entries")
            
            # Process
            id_prepared = prepare_id(id_words)
            print(f"  Processed {len(id_prepared):,} valid words")
            
            # Save
            id_output_path = output_dir / 'dictionary_id.jsonl'
            id_prepared.to_json(id_output_path, orient='records', lines=True, force_ascii=False)
            file_size = id_output_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ Saved to {id_output_path}")
            print(f"    Size: {file_size:.2f} MB")
            print()
            
        except Exception as e:
            print(f"  ✗ Error processing Indonesian dictionary: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    # Process Malaysian dictionary
    if not args.skip_ms:
        ms_input_path = Path(args.ms_input)
        print(f"Processing Malaysian dictionary from {ms_input_path}...")
        
        if not ms_input_path.exists():
            print(f"  ✗ Error: Input file not found: {ms_input_path}")
            print(f"    Please ensure the DBP dictionary JSON file exists.")
            return 1
        
        try:
            # Load Malaysian data
            with open(ms_input_path, 'r', encoding='utf-8') as fp:
                ms_data = json.load(fp)
            
            ms_words = (pd.DataFrame.from_dict(ms_data, orient="index")
                          .rename_axis("word")
                          .reset_index())
            print(f"  Loaded {len(ms_words):,} entries")
            
            # Process
            ms_prepared = prepare_ms(ms_words)
            print(f"  Processed {len(ms_prepared):,} valid words")
            
            # Save
            ms_output_path = output_dir / 'dictionary_ms.jsonl'
            ms_prepared.to_json(ms_output_path, orient='records', lines=True, force_ascii=False)
            file_size = ms_output_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ Saved to {ms_output_path}")
            print(f"    Size: {file_size:.2f} MB")
            print()
            
        except Exception as e:
            print(f"  ✗ Error processing Malaysian dictionary: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print("Generated dictionary files:")
    
    if not args.skip_id:
        id_path = output_dir / 'dictionary_id.jsonl'
        if id_path.exists():
            print(f"  ✓ Indonesian: {id_path}")
    
    if not args.skip_ms:
        ms_path = output_dir / 'dictionary_ms.jsonl'
        if ms_path.exists():
            print(f"  ✓ Malaysian: {ms_path}")
    
    print()
    print("Dictionary format:")
    print("  - word: lowercase word")
    print("  - syllable_list: array of syllable strings")
    print("  - last_syllable: final syllable")
    print("  - last_syllable_index: index of last syllable")
    print()
    print("✓ Dictionary generation complete!")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())