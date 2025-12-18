"""
NLP Processor using syntok, UDPipe, and HuggingFace tokenizers.

This module provides a spaCy-like interface for tokenization, sentence segmentation,
lemmatization, POS tagging, and noun phrase extraction.
"""

import logging
import os
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Try to import dependencies
try:
    import syntok.segmenter as segmenter
    import syntok.tokenizer as tokenizer

    HAS_SYNTOK = True
except ImportError:
    HAS_SYNTOK = False

try:
    from ufal.udpipe import Model, Pipeline

    HAS_UDPIPE = True
except ImportError:
    HAS_UDPIPE = False

try:
    from tokenizers import Tokenizer as HFTokenizer

    HAS_HF_TOKENIZERS = True
except ImportError:
    HAS_HF_TOKENIZERS = False


@dataclass
class Token:
    """Token object compatible with spaCy-like interface."""

    text: str
    lemma: str
    pos: str  # Part of speech tag
    dep: str  # Dependency relation
    idx: int  # Character offset in original text


@dataclass
class Document:
    """Document object containing tokens, sentences, and noun chunks."""

    text: str
    tokens: List[Token]
    sentences: List[str]
    noun_chunks: List[str]
    _udpipe_sentences: List[List[Token]]  # Internal: tokens grouped by sentence


class NLPProcessor:
    """
    NLP processor using syntok, UDPipe, and HuggingFace tokenizers.

    Provides a spaCy-like interface for document processing.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the NLP processor.

        Args:
            model_path: Path to UDPipe model file. If None, will try to download
                       or use default location.
        """
        self._udpipe_model = None
        self._udpipe_pipeline = None
        self._model_path = model_path or self._get_default_model_path()
        self._syntok_tokenizer = tokenizer.Tokenizer() if HAS_SYNTOK else None

    def _get_default_model_path(self) -> str:
        """Get default path for UDPipe English model."""
        # Try to find model in common locations
        home_dir = os.path.expanduser("~")
        model_dir = os.path.join(home_dir, ".udpipe", "models")
        os.makedirs(model_dir, exist_ok=True)

        model_name = "english-ewt-ud-2.5-191206.udpipe"
        model_path = os.path.join(model_dir, model_name)

        # If model doesn't exist, try to download it
        if not os.path.exists(model_path):
            try:
                self._download_model(model_path)
            except Exception as e:
                logger.warning(
                    f"Could not download UDPipe model automatically: {e}. "
                    f"Please download it manually from "
                    f"https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3131/{model_name}"
                )

        return model_path

    def _download_model(self, model_path: str) -> None:
        """Download UDPipe English model if not present."""
        model_url = (
            "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/"
            "handle/11234/1-3131/english-ewt-ud-2.5-191206.udpipe"
        )

        logger.info(f"Downloading UDPipe model from {model_url}...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        urllib.request.urlretrieve(model_url, model_path)
        logger.info(f"Model downloaded to {model_path}")

    def _load_udpipe_model(self) -> None:
        """Lazy load UDPipe model."""
        if not HAS_UDPIPE:
            raise ImportError(
                "ufal.udpipe is not installed. Please install it: pip install ufal.udpipe"
            )

        if self._udpipe_model is None:
            if not os.path.exists(self._model_path):
                raise FileNotFoundError(
                    f"UDPipe model not found at {self._model_path}. "
                    f"Please download it from "
                    f"https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3131/english-ewt-ud-2.5-191206.udpipe"
                )

            self._udpipe_model = Model.load(self._model_path)
            if not self._udpipe_model:
                raise Exception(f"Cannot load UDPipe model from {self._model_path}")

            # Create pipeline: tokenize, tag, parse, output conllu
            self._udpipe_pipeline = Pipeline(
                self._udpipe_model,
                "tokenize",
                Pipeline.DEFAULT,
                Pipeline.DEFAULT,
                "conllu",
            )

    def process(self, text: str) -> Document:
        """
        Process text and return a Document object.

        Args:
            text: Input text to process

        Returns:
            Document object with tokens, sentences, and noun chunks
        """
        if not HAS_SYNTOK:
            raise ImportError(
                "syntok is not installed. Please install it: pip install syntok"
            )

        # Step 1: Use syntok for sentence segmentation
        sentences = self._segment_sentences(text)

        # Step 2: Process with UDPipe for tokenization, POS, lemmatization, parsing
        self._load_udpipe_model()

        all_tokens: List[Token] = []
        udpipe_sentences: List[List[Token]] = []

        for sentence_text in sentences:
            # Process sentence with UDPipe
            processed = self._udpipe_pipeline.process(sentence_text)
            sentence_tokens = self._parse_udpipe_output(processed, text)
            all_tokens.extend(sentence_tokens)
            udpipe_sentences.append(sentence_tokens)

        # Step 3: Extract noun chunks using dependency parsing
        noun_chunks = self._extract_noun_chunks(udpipe_sentences)

        return Document(
            text=text,
            tokens=all_tokens,
            sentences=sentences,
            noun_chunks=noun_chunks,
            _udpipe_sentences=udpipe_sentences,
        )

    def _segment_sentences(self, text: str) -> List[str]:
        """Segment text into sentences using syntok."""
        sentences = []
        for paragraph in segmenter.process(text):
            for sentence in paragraph:
                # Reconstruct sentence text from tokens
                sentence_text = " ".join(token.value for token in sentence)
                sentences.append(sentence_text)
        return sentences

    def _parse_udpipe_output(
        self, conllu_output: str, original_text: str
    ) -> List[Token]:
        """
        Parse UDPipe CoNLL-U output into Token objects.

        Args:
            conllu_output: CoNLL-U formatted output from UDPipe
            original_text: Original text for offset calculation

        Returns:
            List of Token objects
        """
        tokens = []
        char_offset = 0

        for line in conllu_output.strip().split("\n"):
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) < 10:
                continue

            # CoNLL-U format: ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC
            token_id = parts[0]
            # Skip multi-word tokens (IDs like "1-2")
            if "-" in token_id:
                continue

            form = parts[1]  # Surface form
            # Skip empty tokens
            if not form:
                continue

            lemma = parts[2] if parts[2] != "_" else form
            upos = parts[3] if parts[3] != "_" else ""
            deprel = parts[7] if parts[7] != "_" else ""

            # Find character offset in original text
            # Simple approach: find next occurrence after current offset
            idx = original_text.find(form, char_offset)
            if idx == -1:
                idx = char_offset
            else:
                char_offset = idx + len(form)

            token = Token(
                text=form,
                lemma=lemma,
                pos=upos,
                dep=deprel,
                idx=idx,
            )
            tokens.append(token)

        return tokens

    def _extract_noun_chunks(self, sentences: List[List[Token]]) -> List[str]:
        """
        Extract noun phrases from sentences using UDPipe dependency parsing.

        Uses dependency relations to identify noun phrases:
        - Look for noun heads with dependents (determiners, adjectives, other nouns)
        - Patterns: (det)? (amod)* noun+ (nmod)*

        Args:
            sentences: List of sentences, each containing a list of tokens

        Returns:
            List of noun phrase strings
        """
        noun_chunks = []

        for sentence_tokens in sentences:
            if not sentence_tokens:
                continue

            # Build dependency tree structure
            # In CoNLL-U, HEAD is the index of the head token (0-indexed in our list)
            # We need to map token positions to find heads

            # Find noun phrases by looking for:
            # 1. Nouns (NOUN, PROPN) that are heads
            # 2. Sequences of ADJ + NOUN
            # 3. Determiner + Adjective + Noun patterns

            i = 0
            while i < len(sentence_tokens):
                token = sentence_tokens[i]

                # Start of potential noun phrase
                if token.pos in ("NOUN", "PROPN"):
                    # Collect noun phrase starting from this token
                    phrase_tokens = [token]
                    j = i + 1

                    # Look ahead for adjectives, determiners, or other nouns
                    while j < len(sentence_tokens):
                        next_token = sentence_tokens[j]

                        # Stop if we hit punctuation or non-noun-related tokens
                        if next_token.pos in ("PUNCT", "VERB", "ADP", "CCONJ"):
                            break

                        # Include determiners, adjectives, and nouns
                        if next_token.pos in ("DET", "ADJ", "NOUN", "PROPN", "NUM"):
                            phrase_tokens.append(next_token)
                            j += 1
                        else:
                            break

                    # Only add if we have at least one noun
                    if phrase_tokens:
                        phrase_text = " ".join(t.text for t in phrase_tokens)
                        noun_chunks.append(phrase_text)

                    i = j
                else:
                    i += 1

            # Also extract using dependency relations for better quality
            # Look for tokens with dependency relations that indicate noun phrases
            for token in sentence_tokens:
                if token.dep in ("nsubj", "obj", "nmod", "obl") and token.pos in (
                    "NOUN",
                    "PROPN",
                ):
                    # Try to find the full phrase by following dependents
                    phrase = self._get_noun_phrase_from_token(token, sentence_tokens)
                    if phrase and phrase not in noun_chunks:
                        noun_chunks.append(phrase)

        # Deduplicate and filter
        unique_chunks = []
        seen = set()
        for chunk in noun_chunks:
            chunk_lower = chunk.lower().strip()
            if chunk_lower and chunk_lower not in seen:
                seen.add(chunk_lower)
                unique_chunks.append(chunk)

        return unique_chunks

    def _get_noun_phrase_from_token(
        self, head_token: Token, sentence_tokens: List[Token]
    ) -> Optional[str]:
        """
        Get full noun phrase starting from a head token.

        Args:
            head_token: Token that is the head of the noun phrase
            sentence_tokens: All tokens in the sentence

        Returns:
            Noun phrase string or None
        """
        # Simple implementation: collect determiners and adjectives before the noun
        # and nouns/adjectives after it
        phrase_tokens = []

        head_idx = sentence_tokens.index(head_token)

        # Look backwards for determiners and adjectives
        i = head_idx - 1
        while i >= 0:
            token = sentence_tokens[i]
            if token.pos in ("DET", "ADJ", "NUM"):
                phrase_tokens.insert(0, token)
                i -= 1
            else:
                break

        # Add the head
        phrase_tokens.append(head_token)

        # Look forwards for adjectives and nouns (compound nouns)
        i = head_idx + 1
        while i < len(sentence_tokens):
            token = sentence_tokens[i]
            if token.pos in ("ADJ", "NOUN", "PROPN"):
                phrase_tokens.append(token)
                i += 1
            else:
                break

        if len(phrase_tokens) > 0:
            return " ".join(t.text for t in phrase_tokens)
        return None


# Global instance for lazy loading
_nlp_processor: Optional[NLPProcessor] = None


def get_nlp_processor(model_path: Optional[str] = None) -> NLPProcessor:
    """
    Get or create global NLP processor instance.

    Args:
        model_path: Optional path to UDPipe model

    Returns:
        NLPProcessor instance
    """
    global _nlp_processor
    if _nlp_processor is None:
        _nlp_processor = NLPProcessor(model_path=model_path)
    return _nlp_processor
