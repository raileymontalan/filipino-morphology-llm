"""Utilities for splitting Filipino, Indonesian, and Malaysian words into syllables.

Source references:
- Filipino: https://github.com/itudidyay/Tagalog-Word-Syllabization-Python/blob/main/tglSyllabification.py
- Indonesian: https://github.com/fahadh4ilyas/syllable_splitter/blob/master/SyllableSplitter.py
- Malaysian: https://github.com/Kylamber/pemenggalan-kata-indonesia/blob/master/syllabifycation.py
"""

import re
from typing import Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "FilipinoSyllabifier",
    "IndonesianSyllabifier",
    "MalaysianSyllabifier",
]

class FilipinoSyllabifier:
    """Split Filipino (Tagalog) words into syllables."""

    VOWELS = set("AEIOUaeiouÀÁÂÈÉÊÌÍÎÒÓÔÙÚÛàáâèéêìíîòóôùúû")
    LETTER_PAIRS = {"bl", "br", "dr", "pl", "tr"}

    @classmethod
    def _has_vowel(cls, text: str) -> bool:
        """Return True when *text* contains a vowel recognised by the syllabifier."""
        return any(char in cls.VOWELS for char in text)

    @staticmethod
    def _slice_value(source: List[str], index: int, split_at: int) -> List[str]:
        """Split ``source[index]`` into two pieces at ``split_at`` and return a copy."""
        result = source[:]
        result.insert(index + 1, result[index][split_at:])
        result[index] = result[index][:split_at]
        return result

    @staticmethod
    def _merge_values(source: List[str], start: int, end: int) -> List[str]:
        """Join items between *start* and *end* (inclusive) into a single chunk."""
        result = source[:]
        result[start:end + 1] = [''.join(result[start:end + 1])]
        return result

    def syllabify(self, word: str) -> List[str]:
        working = word.replace("'", "")

        for letter in working:
            if letter in self.VOWELS or letter == '-':
                working = working.replace(letter, f" {letter} ")

        working = working.replace('ng', 'ŋ').replace('NG', 'Ŋ')
        segments = working.split()

        offset = 0
        for idx, group in enumerate(list(segments)):
            index = idx + offset
            if index == 0 or index == len(segments) - 1 or segments[index - 1] == '-':
                continue

            group_len = len(group)
            if group_len == 2:
                segments = self._slice_value(segments, index, 1)
                offset += 1
            elif group_len == 3:
                pair = group[1:3].lower()
                if group[0].lower() in {'n', 'm'} and pair in self.LETTER_PAIRS:
                    segments = self._slice_value(segments, index, 1)
                else:
                    segments = self._slice_value(segments, index, 2)
                offset += 1
            elif group_len > 3:
                segments = self._slice_value(segments, index, 2)
                offset += 1

        offset = 0
        snapshot = list(segments)
        for idx, group in enumerate(snapshot):
            if (
                group[-1] in self.VOWELS
                and idx != 0
                and snapshot[idx - 1] not in self.VOWELS
                and snapshot[idx - 1] != '-'
            ):
                segments = self._merge_values(segments, idx - 1 - offset, idx - offset)
                offset += 1

        offset = 0
        snapshot = list(segments)
        for idx, group in enumerate(snapshot):
            if idx == len(snapshot) - 1:
                continue
            next_chunk = snapshot[idx + 1]
            if group[-1] in self.VOWELS and not self._has_vowel(next_chunk) and next_chunk != '-':
                segments = self._merge_values(segments, idx - offset, idx + 1 - offset)
                offset += 1

        result = [seg.replace('ŋ', 'ng').replace('Ŋ', 'NG') for seg in segments if seg != '-']
        return result


class IndonesianSyllabifier:
    """Split Indonesian words into syllables following basic phonetic rules."""

    DEFAULT_CONSONANTS: Tuple[str, ...] = (
        'b', 'c', 'd', 'f', 'g', 'h', 'j',
        'k', 'l', 'm', 'n', 'p', 'q', 'r',
        's', 't', 'v', 'w', 'x', 'y', 'z',
        'ng', 'ny', 'sy', 'ch', 'dh', 'gh',
        'kh', 'ph', 'sh', 'th',
    )
    DEFAULT_DOUBLE_CONSONANTS: Tuple[str, ...] = ('ll', 'ks', 'rs', 'rt')
    DEFAULT_VOWELS: Tuple[str, ...] = ('a', 'e', 'i', 'o', 'u')

    def __init__(
        self,
        consonant: Optional[Sequence[str]] = None,
        vocal: Optional[Sequence[str]] = None,
        double_consonant: Optional[Sequence[str]] = None,
    ) -> None:
        self.consonants = self._merge_with_defaults(self.DEFAULT_CONSONANTS, consonant)
        self.double_consonants = self._merge_with_defaults(
            self.DEFAULT_DOUBLE_CONSONANTS, double_consonant
        )
        self.vowels = self._merge_with_defaults(self.DEFAULT_VOWELS, vocal)

        self._consonant_set = set(self.consonants)
        self._double_consonant_set = set(self.double_consonants)
        self._vowel_set = set(self.vowels)

    @staticmethod
    def _merge_with_defaults(
        defaults: Sequence[str], extras: Optional[Sequence[str]]
    ) -> Tuple[str, ...]:
        items: List[str] = list(defaults)
        if extras:
            for item in extras:
                if item not in items:
                    items.append(item)
        return tuple(items)

    def split_letters(self, word: str) -> Tuple[List[str], str]:
        """Return letters and pattern string describing consonant/vowel layout."""

        letters: List[str] = []
        pattern: List[str] = []
        remaining = word

        while remaining:
            pair = remaining[:2]
            lower_pair = pair.lower()

            if len(pair) == 2 and lower_pair in self._double_consonant_set:
                next_char = remaining[2:3]
                if next_char and next_char.lower() in self._vowel_set:
                    letters.append(pair[0])
                    pattern.append('c')
                    remaining = remaining[1:]
                else:
                    letters.append(pair)
                    pattern.append('c')
                    remaining = remaining[2:]
                continue

            if lower_pair in self._consonant_set:
                letters.append(pair)
                pattern.append('c')
                remaining = remaining[2:]
                continue

            char = remaining[0]
            lower_char = char.lower()

            if lower_char in self._consonant_set:
                letters.append(char)
                pattern.append('c')
            elif lower_char in self._vowel_set:
                letters.append(char)
                pattern.append('v')
            else:
                letters.append(char)
                pattern.append('s')

            remaining = remaining[1:]

        return letters, ''.join(pattern)

    def split_syllables_from_letters(self, letters: List[str], pattern: str) -> List[str]:
        """Insert syllable boundaries based on the consonant/vowel pattern."""

        working_letters = list(letters)
        working_pattern = pattern

        def insert_break(index: int) -> None:
            nonlocal working_letters, working_pattern
            working_letters = working_letters[:index + 1] + ['|'] + working_letters[index + 1:]
            working_pattern = working_pattern[:index + 1] + '|' + working_pattern[index + 1:]

        search = re.search('vc{2,}', working_pattern)
        while search:
            insert_break(search.start() + 1)
            search = re.search('vc{2,}', working_pattern)

        search = re.search(r'v{2,}', working_pattern)
        while search:
            insert_break(search.start())
            search = re.search(r'v{2,}', working_pattern)

        search = re.search(r'vcv', working_pattern)
        while search:
            insert_break(search.start())
            search = re.search(r'vcv', working_pattern)

        search = re.search(r'[cvs]s', working_pattern)
        while search:
            insert_break(search.start())
            search = re.search(r'[cvs]s', working_pattern)

        search = re.search(r's[cvs]', working_pattern)
        while search:
            insert_break(search.start())
            search = re.search(r's[cvs]', working_pattern)

        return ''.join(working_letters).split('|')

    def syllabify(self, word: str) -> List[str]:
        """Return a list of syllables for the provided Indonesian *word*."""

        letters, pattern = self.split_letters(word)
        return self.split_syllables_from_letters(letters, pattern)


class MalaysianSyllabifier:
    """Split Malaysian words into syllables using basic orthographic rules."""

    vocals = list('aiueo*()-')
    diftong = list('*()-')
    gabungan = {
        'kh': '!',
        'ng': '@',
        'sy': '#',
        'ny': '$',
        'tr': '%',
        'gr': '^',
        'ai': '*',
        'ei': '(',
        'au': ')',
        'oi': '-',
    }

    def __init__(self, debug: bool = False) -> None:
        self.debug = debug

    @classmethod
    def replacer(cls, word: str) -> str:
        """Replace multi-character clusters with stand-in symbols."""

        result = word
        for letters, symbol in cls.gabungan.items():
            result = result.replace(letters, symbol)
        return result

    @classmethod
    def unreplace(cls, syllables: Iterable[str]) -> List[str]:
        """Reverse the cluster substitution performed by :meth:`replacer`."""

        result: List[str] = []
        for syllable in syllables:
            expanded = syllable
            for letters, symbol in cls.gabungan.items():
                expanded = expanded.replace(symbol, letters)
            result.append(expanded)
        return result

    @classmethod
    def preprocess(cls, word: str) -> List[str]:
        """Group characters so every consonant is paired with a following vowel."""

        result: List[str] = []
        buffer = ''
        last_consonant = False

        for letter in word:
            is_consonant = letter not in cls.vocals
            if is_consonant:
                buffer += letter
                last_consonant = True
                continue

            if last_consonant:
                if len(buffer) > 1:
                    result.append(buffer[0])
                    result.append(buffer[1:] + letter)
                else:
                    result.append(buffer + letter)
                buffer = ''
            else:
                result.append(letter)
            last_consonant = False

        if buffer:
            result.append(buffer)

        return result

    @classmethod
    def contains(cls, items: Iterable[str], letters: str) -> bool:
        """Return True if *letters* contains any symbol listed in *items*."""

        return any(item in letters for item in items)

    @staticmethod
    def join(letters: Iterable[str]) -> str:
        return ''.join(letters)

    @classmethod
    def process(cls, syllables: List[str]) -> List[str]:
        """Finalize syllable segmentation after preprocessing."""

        result: List[str] = []
        working = list(syllables)

        while True:
            try:
                if not cls.contains(cls.vocals, working[1]):
                    if len(working[1]) == 1:
                        if cls.contains(cls.diftong, working[0]):
                            new_word = cls.join(cls.unreplace([working[0]]))
                            result.append(new_word[:-1])
                            working.insert(0, new_word[-1])
                            del working[1]
                        else:
                            result.append(cls.join(working[:2]))
                            del working[:2]
                    else:
                        result.append(cls.join(working[:2]))
                        del working[:2]
                elif not cls.contains(cls.vocals, working[0]):
                    if len(working) > 2 and not cls.contains(cls.vocals, working[2]):
                        result.append(cls.join(working[:3]))
                        del working[:3]
                    else:
                        result.append(cls.join(working[:2]))
                        del working[:2]
                else:
                    result.append(working[0])
                    del working[0]
            except IndexError:
                if working:
                    result.append(working.pop(0))
                break

        if '%an' in result:
            index = result.index('%an')
            try:
                if result[index + 1][0] == 's' and result[index + 1][1] not in cls.vocals:
                    result[index] = '%ans'
                    result[index + 1] = result[index + 1].replace('s', '')
            except IndexError:
                pass
        elif 'ek' in result:
            index = result.index('ek')
            try:
                if result[index + 1][0] == 's' and result[index + 1][1] not in cls.vocals:
                    result[index] = 'eks'
                    result[index + 1] = result[index + 1].replace('s', '')
            except IndexError:
                pass

        return result

    def syllabify(self, word: str) -> List[str]:
        """Return a list of syllables for the provided Malaysian *word*."""

        replaced_word = self.replacer(word)
        syllables = self.preprocess(replaced_word)
        processed = self.process(syllables)
        result = self.unreplace(processed)

        if self.debug:
            print(result)

        return result