"""
Local + Semantic SEO keywords extraction engine.

This engine uses local NLP (spaCy, YAKE, RAKE, TF-IDF) and semantic processing
(Hugging Face embeddings, zero-shot classification) to extract keywords.
"""

import hashlib
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import spacy

    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False

try:
    from transformers import pipeline as hf_pipeline

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from sentence_transformers import SentenceTransformer

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import yake

    HAS_YAKE = True
except ImportError:
    HAS_YAKE = False

try:
    from rake_nltk import Rake

    HAS_RAKE = True
except ImportError:
    HAS_RAKE = False

try:
    from keyphrase_vectorizers import KeyphraseCountVectorizer

    HAS_KEYPHRASE_VECTORIZER = True
except ImportError:
    HAS_KEYPHRASE_VECTORIZER = False

try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    import fasttext

    HAS_FASTTEXT = True
except ImportError:
    HAS_FASTTEXT = False

try:
    from bertopic import BERTopic

    HAS_BERTOPIC = True
except ImportError:
    HAS_BERTOPIC = False

from marketing_project.models.pipeline_steps import (
    KeywordCluster,
    KeywordDensityAnalysis,
    KeywordMetadata,
    SEOKeywordsResult,
)
from marketing_project.services.engines.base import Engine

logger = logging.getLogger(__name__)


class LocalSemanticSEOKeywordsEngine(Engine):
    """
    Local + Semantic engine for SEO keywords extraction.

    Uses local NLP tools and semantic processing to extract keywords without LLM.
    """

    # Supported operations
    SUPPORTED_OPERATIONS = {
        "extract_main_keyword",
        "extract_primary_keywords",
        "extract_secondary_keywords",
        "extract_lsi_keywords",
        "extract_long_tail_keywords",
        "calculate_density",
        "classify_intent",
        "cluster_keywords",
        "get_seo_metrics",
    }

    def __init__(self):
        """Initialize the local semantic engine."""
        self._nlp = None
        self._embedding_model = None
        self._zero_shot_classifier = None
        self._yake_extractor = None
        self._rake_extractor = None
        self._keyphrase_vectorizer = None
        self._fasttext_model = None
        self._topic_model = None
        # Cache for parsed documents (keyed by content hash)
        # Limited to prevent memory issues (FIFO eviction when limit reached)
        self._parsed_doc_cache: Dict[str, Dict[str, Any]] = {}
        self._parsed_doc_cache_max_size = int(
            os.getenv("LOCAL_SEMANTIC_ENGINE_CACHE_SIZE", "50")
        )

    def _get_nlp(self):
        """Lazy load spaCy model."""
        if not HAS_SPACY:
            raise ImportError(
                "spacy is not installed. Please install it: pip install spacy && python -m spacy download en_core_web_sm"
            )
        if self._nlp is None:
            try:
                import spacy

                self._nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.error(
                    "spaCy model 'en_core_web_sm' not found. "
                    "Please run: python -m spacy download en_core_web_sm"
                )
                raise
        return self._nlp

    def _get_embedding_model(self):
        """Lazy load sentence transformer model."""
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError(
                "sentence-transformers is not installed. Please install it: pip install sentence-transformers"
            )
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._embedding_model = SentenceTransformer(
                    "all-MiniLM-L6-v2"
                )  # Lightweight model
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise
        return self._embedding_model

    def _get_zero_shot_classifier(self):
        """Lazy load zero-shot classifier."""
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers is not installed. Please install it: pip install transformers"
            )
        if self._zero_shot_classifier is None:
            try:
                from transformers import pipeline as hf_pipeline

                self._zero_shot_classifier = hf_pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli",
                )
            except Exception as e:
                logger.error(f"Failed to load zero-shot classifier: {e}")
                raise
        return self._zero_shot_classifier

    def _get_yake_extractor(self):
        """Lazy load YAKE extractor."""
        if not HAS_YAKE:
            raise ImportError(
                "yake is not installed. Please install it: pip install yake"
            )
        if self._yake_extractor is None:
            try:
                import yake

                self._yake_extractor = yake.KeywordExtractor(
                    lan="en", n=3, dedupLim=0.7, top=100
                )
            except Exception as e:
                logger.error(f"Failed to initialize YAKE: {e}")
                raise
        return self._yake_extractor

    def _get_rake_extractor(self):
        """Lazy load RAKE extractor."""
        if not HAS_RAKE:
            raise ImportError(
                "rake-nltk is not installed. Please install it: pip install rake-nltk"
            )
        if self._rake_extractor is None:
            try:
                from rake_nltk import Rake

                self._rake_extractor = Rake()
            except Exception as e:
                logger.error(f"Failed to initialize RAKE: {e}")
                raise
        return self._rake_extractor

    def _get_keyphrase_vectorizer(self):
        """Lazy load KeyPhrase-BERT vectorizer."""
        if not HAS_KEYPHRASE_VECTORIZER:
            logger.warning(
                "keyphrase-vectorizers not available, skipping KeyPhrase-BERT"
            )
            return None
        if self._keyphrase_vectorizer is None:
            try:
                from keyphrase_vectorizers import KeyphraseCountVectorizer

                self._keyphrase_vectorizer = KeyphraseCountVectorizer()
            except Exception as e:
                logger.warning(f"Failed to initialize KeyPhrase-BERT: {e}")
                return None
        return self._keyphrase_vectorizer

    def supports_operation(self, operation: str) -> bool:
        """Check if operation is supported."""
        return operation in self.SUPPORTED_OPERATIONS

    async def execute(
        self,
        operation: str,
        inputs: Dict[str, Any],
        context: Dict[str, Any],
        pipeline: Optional[Any] = None,
    ) -> Any:
        """
        Execute an operation.

        Args:
            operation: Operation name
            inputs: Input data (must contain 'content' dict)
            context: Execution context
            pipeline: Not used for local engine

        Returns:
            Operation result
        """
        if operation not in self.SUPPORTED_OPERATIONS:
            raise ValueError(f"Unsupported operation: {operation}")

        # Check if parsed_doc is already available (from previous operation)
        parsed_doc = inputs.get("_parsed_doc")

        if parsed_doc is None:
            # Need to parse - get content
            content = inputs.get("content", {})
            if isinstance(content, dict):
                content_str = content.get("content", "")
                title = content.get("title", "")
            else:
                content_str = str(content)
                title = ""

            # Preprocess document once (cached)
            parsed_doc = self._preprocess_document_cached(content_str, title, context)

        # Store parsed_doc in inputs so dependent operations can reuse it
        inputs["_parsed_doc"] = parsed_doc

        # Route to appropriate operation
        if operation == "extract_main_keyword":
            return await self._extract_main_keyword(parsed_doc, inputs, context)
        elif operation == "extract_primary_keywords":
            return await self._extract_primary_keywords(parsed_doc, inputs, context)
        elif operation == "extract_secondary_keywords":
            return await self._extract_secondary_keywords(parsed_doc, inputs, context)
        elif operation == "extract_lsi_keywords":
            return await self._extract_lsi_keywords(parsed_doc, inputs, context)
        elif operation == "extract_long_tail_keywords":
            return await self._extract_long_tail_keywords(parsed_doc, inputs, context)
        elif operation == "calculate_density":
            return await self._calculate_density(parsed_doc, inputs, context)
        elif operation == "classify_intent":
            return await self._classify_intent(parsed_doc, inputs, context)
        elif operation == "cluster_keywords":
            return await self._cluster_keywords(parsed_doc, inputs, context)
        elif operation == "get_seo_metrics":
            return await self._get_seo_metrics(parsed_doc, inputs, context)
        else:
            raise ValueError(f"Operation not implemented: {operation}")

    def _get_content_hash(self, content: str, title: str) -> str:
        """Generate hash for content to use as cache key."""
        combined = f"{title}|||{content}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _preprocess_document_cached(
        self, content: str, title: str = "", context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Preprocess document with spaCy, using cache if available.

        Args:
            content: Content text
            title: Title text
            context: Execution context (may contain pre-extracted headings)

        Returns:
            Dict with parsed document data
        """
        # Check cache first
        content_hash = self._get_content_hash(content, title)
        if content_hash in self._parsed_doc_cache:
            logger.debug(
                f"Using cached parsed document for content hash: {content_hash[:8]}"
            )
            return self._parsed_doc_cache[content_hash]

        # Check if headings are already extracted (from preprocessing/context)
        headings = None
        if context:
            # Check for headings in context from previous steps
            content_structure = context.get("content_structure", {})
            if content_structure:
                headings = []
                headings.extend(content_structure.get("h1_headings", []))
                headings.extend(content_structure.get("h2_headings", []))
                headings.extend(content_structure.get("h3_headings", []))

        # Parse with spaCy
        nlp = self._get_nlp()
        doc = nlp(content)
        title_doc = nlp(title) if title else None

        # Extract headings if not already available
        if headings is None:
            headings = self._extract_headings(content)

        parsed_doc = {
            "doc": doc,
            "title_doc": title_doc,
            "title": title,
            "headings": headings,
            "body_text": content,
            "tokens": [token for token in doc],
            "noun_chunks": [chunk.text for chunk in doc.noun_chunks],
            "sentences": [sent.text for sent in doc.sents],
        }

        # Cache the parsed document (with size limit)
        if len(self._parsed_doc_cache) >= self._parsed_doc_cache_max_size:
            # Remove oldest entry (FIFO eviction)
            oldest_key = next(iter(self._parsed_doc_cache))
            del self._parsed_doc_cache[oldest_key]
        self._parsed_doc_cache[content_hash] = parsed_doc
        logger.debug(f"Cached parsed document for content hash: {content_hash[:8]}")

        return parsed_doc

    def _preprocess_document(self, content: str, title: str = "") -> Dict[str, Any]:
        """
        Preprocess document with spaCy (non-cached version for backward compatibility).

        Args:
            content: Content text
            title: Title text

        Returns:
            Dict with parsed document data
        """
        return self._preprocess_document_cached(content, title, None)

    def _extract_headings(self, content: str) -> List[str]:
        """Extract headings from content."""
        headings = []
        patterns = [
            (r"<h1[^>]*>(.*?)</h1>", 1),
            (r"<h2[^>]*>(.*?)</h2>", 2),
            (r"<h3[^>]*>(.*?)</h3>", 3),
            (r"^#\s+(.+)$", 1),
            (r"^##\s+(.+)$", 2),
            (r"^###\s+(.+)$", 3),
        ]

        for pattern, level in patterns:
            matches = re.findall(pattern, content, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                heading_text = match[0] if isinstance(match, tuple) else match
                headings.append(heading_text.strip())

        return headings

    def _generate_candidates(self, parsed_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate candidate keywords from multiple sources.

        Returns:
            List of candidate dicts with text, scores, and features
        """
        candidates = []
        text = parsed_doc["body_text"]
        title = parsed_doc["title"]
        headings = parsed_doc["headings"]
        full_text = f"{title} {' '.join(headings)} {text}"

        # YAKE extraction
        try:
            yake_extractor = self._get_yake_extractor()
            yake_keywords = yake_extractor.extract_keywords(full_text)
            for score, phrase in yake_keywords:
                if 1 <= len(phrase.split()) <= 5:  # 1-5 words
                    candidates.append(
                        {
                            "text": phrase,
                            "yake_score": score,
                            "source": "yake",
                        }
                    )
        except Exception as e:
            logger.warning(f"YAKE extraction failed: {e}")

        # RAKE extraction
        try:
            rake_extractor = self._get_rake_extractor()
            rake_extractor.extract_keywords_from_text(full_text)
            rake_keywords = rake_extractor.get_ranked_phrases()
            for i, phrase in enumerate(rake_keywords[:50]):  # Top 50
                candidates.append(
                    {
                        "text": phrase,
                        "rake_score": 1.0 / (i + 1),  # Inverse rank
                        "source": "rake",
                    }
                )
        except Exception as e:
            logger.warning(f"RAKE extraction failed: {e}")

        # TF-IDF extraction
        try:
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 3), max_features=100, stop_words="english"
            )
            vectorizer.fit([full_text])
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = vectorizer.transform([full_text]).toarray()[0]

            for phrase, score in zip(feature_names, tfidf_scores):
                if score > 0:
                    candidates.append(
                        {
                            "text": phrase,
                            "tfidf_score": float(score),
                            "source": "tfidf",
                        }
                    )
        except Exception as e:
            logger.warning(f"TF-IDF extraction failed: {e}")

        # KeyPhrase-BERT extraction (Phase 1.1)
        try:
            kp_vectorizer = self._get_keyphrase_vectorizer()
            if kp_vectorizer:
                kp_vectorizer.fit([full_text])
                kp_keywords = kp_vectorizer.get_feature_names_out()
                # Get TF-IDF scores for keyphrases
                tfidf_for_kp = TfidfVectorizer(ngram_range=(1, 5), stop_words="english")
                tfidf_for_kp.fit([full_text])
                kp_scores = tfidf_for_kp.transform([full_text]).toarray()[0]
                kp_feature_names = tfidf_for_kp.get_feature_names_out()

                for phrase in kp_keywords[:50]:  # Top 50
                    if phrase in kp_feature_names:
                        idx = list(kp_feature_names).index(phrase)
                        score = float(kp_scores[idx])
                    else:
                        score = 0.5  # Default score

                    candidates.append(
                        {
                            "text": phrase,
                            "kbert_score": score,
                            "source": "kbert",
                        }
                    )
        except Exception as e:
            logger.warning(f"KeyPhrase-BERT extraction failed: {e}")

        # PhraseRank/PositionRank extraction (Phase 1.2)
        try:
            if HAS_NETWORKX:
                graph_candidates = self._extract_with_phraserank(full_text, parsed_doc)
                candidates.extend(graph_candidates)
        except Exception as e:
            logger.warning(f"PhraseRank extraction failed: {e}")

        # Merge and normalize candidates
        return self._merge_candidates(candidates, parsed_doc)

    def _extract_with_phraserank(
        self, text: str, parsed_doc: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extract keyphrases using graph-based PhraseRank/PositionRank.

        Args:
            text: Full text
            parsed_doc: Parsed document data

        Returns:
            List of candidate dicts with phraserank scores
        """
        if not HAS_NETWORKX:
            return []

        from collections import defaultdict

        import networkx as nx

        # Build co-occurrence graph
        sentences = parsed_doc.get("sentences", [text])
        graph = nx.Graph()
        phrase_positions = defaultdict(list)

        # Extract noun phrases and build graph
        nlp = self._get_nlp()
        for sent_idx, sentence in enumerate(sentences[:50]):  # Limit for performance
            doc = nlp(sentence)
            phrases = [chunk.text.lower().strip() for chunk in doc.noun_chunks]

            # Add phrases as nodes
            for phrase in phrases:
                if len(phrase.split()) <= 5:  # 1-5 words
                    graph.add_node(phrase)
                    phrase_positions[phrase].append(sent_idx)

            # Add edges between co-occurring phrases
            for i, phrase1 in enumerate(phrases):
                for phrase2 in phrases[i + 1 :]:
                    if phrase1 != phrase2:
                        if graph.has_edge(phrase1, phrase2):
                            graph[phrase1][phrase2]["weight"] += 1
                        else:
                            graph.add_edge(phrase1, phrase2, weight=1)

        if len(graph.nodes()) == 0:
            return []

        # Compute PageRank
        try:
            pagerank_scores = nx.pagerank(graph, weight="weight")
        except:
            pagerank_scores = {node: 1.0 for node in graph.nodes()}

        # Compute position-based scores (earlier = better)
        position_scores = {}
        for phrase, positions in phrase_positions.items():
            # Earlier positions get higher scores
            avg_position = sum(positions) / len(positions) if positions else 0
            position_scores[phrase] = 1.0 / (1.0 + avg_position / 10.0)

        # Combine scores
        candidates = []
        for phrase in graph.nodes():
            pr_score = pagerank_scores.get(phrase, 0)
            pos_score = position_scores.get(phrase, 0)
            combined_score = pr_score * 0.6 + pos_score * 0.4

            candidates.append(
                {
                    "text": phrase,
                    "phraserank_score": combined_score,
                    "source": "phraserank",
                }
            )

        # Sort by score and return top candidates
        candidates.sort(key=lambda x: x["phraserank_score"], reverse=True)
        return candidates[:30]  # Top 30

    def _merge_candidates(
        self, candidates: List[Dict[str, Any]], parsed_doc: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Merge candidates from different sources, deduplicate, and add features.

        Args:
            candidates: List of candidate dicts
            parsed_doc: Parsed document data

        Returns:
            Merged and normalized candidate list
        """
        # Group by normalized text
        candidate_map = {}
        for cand in candidates:
            normalized = self._normalize_keyword(cand["text"])
            if normalized not in candidate_map:
                candidate_map[normalized] = {
                    "text": cand["text"],
                    "yake_score": 0.0,
                    "rake_score": 0.0,
                    "tfidf_score": 0.0,
                    "frequency": 0,
                }

            # Merge scores
            if "yake_score" in cand:
                candidate_map[normalized]["yake_score"] = cand["yake_score"]
            if "rake_score" in cand:
                candidate_map[normalized]["rake_score"] = cand["rake_score"]
            if "tfidf_score" in cand:
                candidate_map[normalized]["tfidf_score"] = cand["tfidf_score"]
            if "kbert_score" in cand:
                candidate_map[normalized]["kbert_score"] = cand["kbert_score"]
            if "phraserank_score" in cand:
                candidate_map[normalized]["phraserank_score"] = cand["phraserank_score"]

        # Add features
        merged = []
        title_lower = parsed_doc["title"].lower()
        headings_lower = [h.lower() for h in parsed_doc["headings"]]
        body_lower = parsed_doc["body_text"].lower()

        for normalized, cand in candidate_map.items():
            text = cand["text"]
            text_lower = text.lower()

            # Count frequency
            cand["frequency"] = body_lower.count(text_lower)

            # Check placement
            cand["in_title"] = text_lower in title_lower
            cand["in_headings"] = any(text_lower in h for h in headings_lower)

            # Add length
            cand["length"] = len(text.split())

            merged.append(cand)

        # Initialize scores for new sources if missing
        for cand in merged:
            if "kbert_score" not in cand:
                cand["kbert_score"] = 0.0
            if "phraserank_score" not in cand:
                cand["phraserank_score"] = 0.0

        # Sort by combined importance (updated weights for new sources)
        merged.sort(
            key=lambda x: (
                x.get("yake_score", 0) * 0.25
                + x.get("tfidf_score", 0) * 0.20
                + x.get("rake_score", 0) * 0.15
                + x.get("kbert_score", 0) * 0.20
                + x.get("phraserank_score", 0) * 0.15
                + (1.0 if x["in_title"] else 0.0) * 0.05
            ),
            reverse=True,
        )

        return merged

    def _normalize_keyword(self, keyword: str) -> str:
        """Normalize keyword for comparison."""
        return keyword.strip().lower()

    def _get_fasttext_model(self):
        """Lazy load FastText model for semantic neighbors."""
        if not HAS_FASTTEXT:
            logger.warning("fasttext not available, skipping semantic neighbors")
            return None
        if self._fasttext_model is None:
            try:
                import fasttext

                # Use pretrained model (downloads automatically)
                # Using a lightweight model for performance
                self._fasttext_model = fasttext.load_model("cc.en.300.bin")
            except Exception as e:
                logger.warning(f"Failed to load FastText model: {e}")
                return None
        return self._fasttext_model

    def _get_semantic_neighbors(
        self, keyword: str, top_n: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Get semantic neighbors using FastText/Word2Vec (Phase 2.1).

        Args:
            keyword: Keyword to find neighbors for
            top_n: Number of neighbors to return

        Returns:
            List of (neighbor, similarity_score) tuples
        """
        model = self._get_fasttext_model()
        if model is None:
            return []

        try:
            # Get word vectors
            words = keyword.lower().split()
            if len(words) == 1:
                # Single word - get nearest neighbors
                neighbors = model.get_nearest_neighbors(keyword.lower(), k=top_n)
                return [(word, score) for score, word in neighbors]
            else:
                # Multi-word phrase - average word vectors
                word_vectors = []
                for word in words:
                    try:
                        vec = model.get_word_vector(word)
                        word_vectors.append(vec)
                    except:
                        continue

                if not word_vectors:
                    return []

                # Average vectors
                avg_vector = np.mean(word_vectors, axis=0)

                # Find nearest neighbors (simplified - would need full vocabulary search)
                # For now, return neighbors of component words
                all_neighbors = []
                for word in words:
                    try:
                        neighbors = model.get_nearest_neighbors(word, k=top_n)
                        all_neighbors.extend([(w, s) for s, w in neighbors])
                    except:
                        continue

                # Deduplicate and sort
                seen = set()
                unique_neighbors = []
                for word, score in all_neighbors:
                    if word not in seen and word not in words:
                        seen.add(word)
                        unique_neighbors.append((word, score))

                return sorted(unique_neighbors, key=lambda x: x[1], reverse=True)[
                    :top_n
                ]
        except Exception as e:
            logger.warning(f"Failed to get semantic neighbors: {e}")
            return []

    def _build_cooccurrence_graph(
        self, parsed_doc: Dict[str, Any], candidates: List[Dict[str, Any]]
    ) -> Optional[Any]:
        """
        Build co-occurrence graph using networkx (Phase 2.2).

        Args:
            parsed_doc: Parsed document data
            candidates: Candidate keywords

        Returns:
            NetworkX graph or None
        """
        if not HAS_NETWORKX:
            return None

        import networkx as nx

        graph = nx.Graph()
        sentences = parsed_doc.get("sentences", [])
        candidate_texts = {self._normalize_keyword(c["text"]): c for c in candidates}

        # Build graph from sentence co-occurrence
        for sentence in sentences[:100]:  # Limit for performance
            sentence_lower = sentence.lower()
            found_keywords = [
                kw for kw in candidate_texts.keys() if kw in sentence_lower
            ]

            # Add edges between co-occurring keywords
            for i, kw1 in enumerate(found_keywords):
                graph.add_node(kw1)
                for kw2 in found_keywords[i + 1 :]:
                    graph.add_node(kw2)
                    if graph.has_edge(kw1, kw2):
                        graph[kw1][kw2]["weight"] += 1
                    else:
                        graph.add_edge(kw1, kw2, weight=1)

        return graph if len(graph.nodes()) > 0 else None

    async def _extract_main_keyword(
        self,
        parsed_doc: Dict[str, Any],
        inputs: Dict[str, Any],
        context: Dict[str, Any],
    ) -> str:
        """Extract main keyword (1-3 words, highest importance)."""
        candidates = self._generate_candidates(parsed_doc)

        # Filter to 1-3 words
        filtered = [c for c in candidates if 1 <= c["length"] <= 3]

        if not filtered:
            # Fallback to first candidate
            return candidates[0]["text"] if candidates else "keyword"

        # Get embeddings for top candidates
        embedding_model = self._get_embedding_model()
        doc_embedding = embedding_model.encode([parsed_doc["body_text"]])[0]

        # Score candidates with semantic similarity
        top_candidates = filtered[:20]  # Top 20 for efficiency
        candidate_texts = [c["text"] for c in top_candidates]
        candidate_embeddings = embedding_model.encode(candidate_texts)

        similarities = cosine_similarity([doc_embedding], candidate_embeddings)[0]

        # Combine scores
        for i, cand in enumerate(top_candidates):
            semantic_score = float(similarities[i])
            cand["combined_score"] = (
                cand.get("yake_score", 0) * 0.3
                + cand.get("tfidf_score", 0) * 0.3
                + semantic_score * 0.3
                + (1.0 if cand["in_title"] else 0.0) * 0.1
            )

        # Return top candidate
        top_candidate = max(top_candidates, key=lambda x: x["combined_score"])
        return top_candidate["text"]

    async def _extract_primary_keywords(
        self,
        parsed_doc: Dict[str, Any],
        inputs: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[str]:
        """Extract primary keywords (3-5 total, including main)."""
        main_keyword = await self._extract_main_keyword(parsed_doc, inputs, context)
        candidates = self._generate_candidates(parsed_doc)

        # Filter to 1-3 words, exclude main keyword
        filtered = [
            c
            for c in candidates
            if 1 <= c["length"] <= 3
            and self._normalize_keyword(c["text"])
            != self._normalize_keyword(main_keyword)
        ]

        # Get embeddings
        embedding_model = self._get_embedding_model()
        doc_embedding = embedding_model.encode([parsed_doc["body_text"]])[0]
        main_embedding = embedding_model.encode([main_keyword])[0]

        top_candidates = filtered[:30]
        candidate_texts = [c["text"] for c in top_candidates]
        candidate_embeddings = embedding_model.encode(candidate_texts)

        doc_similarities = cosine_similarity([doc_embedding], candidate_embeddings)[0]
        main_similarities = cosine_similarity([main_embedding], candidate_embeddings)[0]

        # Score and deduplicate
        scored = []
        seen_embeddings = [main_embedding]

        for i, cand in enumerate(top_candidates):
            text_embedding = candidate_embeddings[i]

            # Check for near-duplicates
            if any(
                cosine_similarity([text_embedding], [seen])[0][0] > 0.85
                for seen in seen_embeddings
            ):
                continue

            semantic_score = float(doc_similarities[i])
            main_similarity = float(main_similarities[i])

            cand["combined_score"] = (
                cand.get("yake_score", 0) * 0.25
                + cand.get("tfidf_score", 0) * 0.25
                + semantic_score * 0.25
                + main_similarity * 0.15
                + (1.0 if cand["in_title"] else 0.0) * 0.1
            )

            scored.append(cand)
            seen_embeddings.append(text_embedding)

        # Sort and take top 4 (plus main = 5 total)
        scored.sort(key=lambda x: x["combined_score"], reverse=True)
        primary = [main_keyword] + [c["text"] for c in scored[:4]]

        return primary[:5]  # Ensure max 5

    async def _extract_secondary_keywords(
        self,
        parsed_doc: Dict[str, Any],
        inputs: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[str]:
        """Extract secondary keywords (5-10, supporting keywords)."""
        # Get primary keywords to exclude
        primary = await self._extract_primary_keywords(parsed_doc, inputs, context)
        primary_normalized = {self._normalize_keyword(kw) for kw in primary}

        candidates = self._generate_candidates(parsed_doc)

        # Phase 2.1: Add semantic neighbors of primary keywords
        semantic_candidates = []
        for primary_kw in primary[:3]:  # Top 3 primaries
            neighbors = self._get_semantic_neighbors(primary_kw, top_n=5)
            for neighbor_word, similarity in neighbors:
                neighbor_normalized = self._normalize_keyword(neighbor_word)
                if neighbor_normalized not in primary_normalized:
                    # Check if neighbor appears in document
                    if neighbor_word.lower() in parsed_doc["body_text"].lower():
                        semantic_candidates.append(
                            {
                                "text": neighbor_word,
                                "semantic_similarity": similarity,
                                "source": "fasttext_neighbor",
                                "frequency": parsed_doc["body_text"]
                                .lower()
                                .count(neighbor_word.lower()),
                            }
                        )

        # Add semantic candidates to pool
        candidates.extend(semantic_candidates)

        # Filter: exclude primaries, 1-4 words
        filtered = [
            c
            for c in candidates
            if 1 <= c["length"] <= 4
            and self._normalize_keyword(c["text"]) not in primary_normalized
        ]

        # Get embeddings
        embedding_model = self._get_embedding_model()
        doc_embedding = embedding_model.encode([parsed_doc["body_text"]])[0]

        top_candidates = filtered[:40]
        candidate_texts = [c["text"] for c in top_candidates]
        candidate_embeddings = embedding_model.encode(candidate_texts)

        similarities = cosine_similarity([doc_embedding], candidate_embeddings)[0]

        # Score
        for i, cand in enumerate(top_candidates):
            semantic_score = float(similarities[i])
            cand["combined_score"] = (
                cand.get("yake_score", 0) * 0.3
                + cand.get("tfidf_score", 0) * 0.3
                + semantic_score * 0.4
            )

        # Sort and take top 10
        top_candidates.sort(key=lambda x: x["combined_score"], reverse=True)
        return [c["text"] for c in top_candidates[:10]]

    async def _extract_lsi_keywords(
        self,
        parsed_doc: Dict[str, Any],
        inputs: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[str]:
        """Extract LSI keywords (semantic neighbors of main keyword)."""
        main_keyword = await self._extract_main_keyword(parsed_doc, inputs, context)
        primary = await self._extract_primary_keywords(parsed_doc, inputs, context)
        secondary = await self._extract_secondary_keywords(parsed_doc, inputs, context)

        exclude = {
            self._normalize_keyword(kw) for kw in [main_keyword] + primary + secondary
        }

        candidates = self._generate_candidates(parsed_doc)
        filtered = [
            c for c in candidates if self._normalize_keyword(c["text"]) not in exclude
        ]

        # Get embeddings
        embedding_model = self._get_embedding_model()
        main_embedding = embedding_model.encode([main_keyword])[0]

        top_candidates = filtered[:50]
        candidate_texts = [c["text"] for c in top_candidates]
        candidate_embeddings = embedding_model.encode(candidate_texts)

        similarities = cosine_similarity([main_embedding], candidate_embeddings)[0]

        # Score by similarity to main
        for i, cand in enumerate(top_candidates):
            cand["similarity"] = float(similarities[i])

        # Sort and take top 10
        top_candidates.sort(key=lambda x: x["similarity"], reverse=True)
        return [c["text"] for c in top_candidates[:10]]

    def _is_query_like_phrase(self, phrase: str) -> bool:
        """
        Check if phrase looks like a query (Phase 5.2).

        Args:
            phrase: Phrase to check

        Returns:
            True if phrase looks query-like
        """
        phrase_lower = phrase.lower()
        query_indicators = [
            "how",
            "what",
            "when",
            "where",
            "why",
            "which",
            "who",
            "best",
            "top",
            "vs",
            "versus",
            "for",
            "to",
            "guide",
            "tutorial",
            "review",
            "compare",
            "difference",
            "between",
            "near me",
            "cost",
            "price",
            "free",
            "online",
            "2024",
            "2025",
        ]
        return any(indicator in phrase_lower for indicator in query_indicators)

    async def _extract_long_tail_keywords(
        self,
        parsed_doc: Dict[str, Any],
        inputs: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[str]:
        """Extract long-tail keywords (3-6 words, query-like) - Phase 5.2 enhanced."""
        primary = await self._extract_primary_keywords(parsed_doc, inputs, context)
        primary_terms = set()
        for kw in primary:
            primary_terms.update(kw.lower().split())

        candidates = self._generate_candidates(parsed_doc)

        # Phase 5.2: Enhanced query-like phrase mining
        # Filter to 3-6 words, prefer query-like
        filtered = [
            c
            for c in candidates
            if 3 <= c["length"] <= 6
            and (
                any(term in c["text"].lower() for term in primary_terms)
                or self._is_query_like_phrase(c["text"])
            )
        ]

        # Prefer query-like shapes (Phase 5.2)
        query_like = [c for c in filtered if self._is_query_like_phrase(c["text"])]

        # Get embeddings
        embedding_model = self._get_embedding_model()
        doc_embedding = embedding_model.encode([parsed_doc["body_text"]])[0]

        candidates_to_score = (
            query_like[:20] + [c for c in filtered if c not in query_like][:20]
        )
        candidate_texts = [c["text"] for c in candidates_to_score]
        candidate_embeddings = embedding_model.encode(candidate_texts)

        similarities = cosine_similarity([doc_embedding], candidate_embeddings)[0]

        # Score with Phase 5.2 query-like boost
        for i, cand in enumerate(candidates_to_score):
            semantic_score = float(similarities[i])
            is_query_like = self._is_query_like_phrase(cand["text"])
            query_boost = 1.3 if is_query_like else 1.0  # Higher boost for query-like
            cand["combined_score"] = (
                cand.get("tfidf_score", 0) * 0.4 + semantic_score * 0.6
            ) * query_boost

        # Sort and take top 8, prioritizing query-like
        candidates_to_score.sort(key=lambda x: x["combined_score"], reverse=True)
        result = [c["text"] for c in candidates_to_score[:8]]

        # Ensure at least 3 query-like phrases if available
        query_like_results = [kw for kw in result if self._is_query_like_phrase(kw)]
        if len(query_like_results) < 3:
            # Add more query-like from remaining candidates
            remaining_query_like = [
                c["text"]
                for c in candidates_to_score[8:]
                if self._is_query_like_phrase(c["text"]) and c["text"] not in result
            ]
            result.extend(remaining_query_like[: 3 - len(query_like_results)])
            result = result[:8]  # Keep max 8

        return result

    async def _calculate_density(
        self,
        parsed_doc: Dict[str, Any],
        inputs: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[KeywordDensityAnalysis]:
        """Calculate keyword density analysis."""
        # Get keywords from inputs or extract
        primary = inputs.get("primary_keywords", [])
        secondary = inputs.get("secondary_keywords", [])

        if not primary:
            primary = await self._extract_primary_keywords(parsed_doc, inputs, context)

        keywords = primary + secondary
        if not keywords:
            return []

        content = parsed_doc["body_text"]
        content_lower = content.lower()
        total_words = len(content.split())

        analyses = []
        nlp = self._get_nlp()

        for keyword in keywords:
            keyword_lower = keyword.lower()
            keyword_words = keyword.split()

            # Count exact occurrences
            exact_count = content_lower.count(keyword_lower)

            # Count lemma-based occurrences
            keyword_doc = nlp(keyword)
            keyword_lemmas = {token.lemma_.lower() for token in keyword_doc}
            lemma_count = sum(
                1
                for token in parsed_doc["doc"]
                if token.lemma_.lower() in keyword_lemmas
            )

            occurrences = max(exact_count, lemma_count // len(keyword_words))

            # Calculate density
            keyword_word_count = len(keyword_words)
            current_density = (
                (occurrences * keyword_word_count / total_words * 100)
                if total_words > 0
                else 0.0
            )

            # Optimal density
            optimal_density = 2.0 if keyword in primary[:5] else 0.75

            # Placement locations
            placement_locations = []
            if keyword_lower in content_lower[:200]:
                placement_locations.append("introduction")

            if any(keyword_lower in h.lower() for h in parsed_doc["headings"]):
                placement_locations.append("heading")

            if not placement_locations:
                placement_locations.append("body")

            analyses.append(
                KeywordDensityAnalysis(
                    keyword=keyword,
                    current_density=round(current_density, 2),
                    optimal_density=optimal_density,
                    occurrences=occurrences,
                    placement_locations=placement_locations,
                )
            )

        return analyses

    async def _classify_intent(
        self,
        parsed_doc: Dict[str, Any],
        inputs: Dict[str, Any],
        context: Dict[str, Any],
    ) -> str:
        """Classify search intent using zero-shot classification."""
        main_keyword = inputs.get("main_keyword")
        if not main_keyword:
            main_keyword = await self._extract_main_keyword(parsed_doc, inputs, context)

        classifier = self._get_zero_shot_classifier()
        labels = ["informational", "transactional", "navigational", "commercial"]

        try:
            result = classifier(main_keyword, labels)
            return result["labels"][0]  # Top label
        except Exception as e:
            logger.warning(
                f"Intent classification failed: {e}, defaulting to 'informational'"
            )
            return "informational"

    def _get_topic_model(self, documents: List[str]):
        """
        Get or create BERTopic model (Phase 3.1).

        Args:
            documents: List of documents to model

        Returns:
            BERTopic model or None
        """
        if not HAS_BERTOPIC:
            logger.warning("bertopic not available, skipping topic modeling")
            return None

        try:
            from bertopic import BERTopic

            # Create new model for each document set (can be optimized later)
            # Initialize with minimal config for performance
            topic_model = BERTopic(
                verbose=False,
                calculate_probabilities=False,
                nr_topics="auto",
            )
            return topic_model
        except Exception as e:
            logger.warning(f"Failed to initialize BERTopic: {e}")
            return None

    async def _cluster_keywords(
        self,
        parsed_doc: Dict[str, Any],
        inputs: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[KeywordCluster]:
        """Cluster keywords by semantic similarity with topic modeling (Phase 3)."""
        # Get all keywords
        primary = inputs.get("primary_keywords", [])
        secondary = inputs.get("secondary_keywords", [])
        lsi = inputs.get("lsi_keywords", [])
        long_tail = inputs.get("long_tail_keywords", [])

        all_keywords = primary + secondary + lsi + long_tail
        if len(all_keywords) < 3:
            return []

        # Phase 2.2: Build co-occurrence graph
        candidates = self._generate_candidates(parsed_doc)
        cooccurrence_graph = self._build_cooccurrence_graph(parsed_doc, candidates)

        # Get embeddings
        embedding_model = self._get_embedding_model()
        embeddings = embedding_model.encode(all_keywords)

        # Phase 3.1: Try topic modeling first
        topic_labels = None
        topic_info = None
        try:
            topic_model = self._get_topic_model([parsed_doc["body_text"]])
            if topic_model:
                # Fit model on document
                topics, probs = topic_model.fit_transform([parsed_doc["body_text"]])
                # Map keywords to topics (simplified - use document topic)
                # In a full implementation, would transform each keyword
                doc_topic = topics[0] if len(topics) > 0 else -1
                topic_labels = np.array([doc_topic] * len(all_keywords))
                try:
                    topic_info = topic_model.get_topic_info()
                except:
                    topic_info = None
        except Exception as e:
            logger.warning(f"Topic modeling failed: {e}")

        # Cluster using embeddings
        n_clusters = min(5, len(all_keywords) // 2)
        if n_clusters < 2:
            n_clusters = 2

        try:
            clustering = AgglomerativeClustering(n_clusters=n_clusters)
            cluster_labels = clustering.fit_predict(embeddings)
        except Exception:
            clustering = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = clustering.fit_predict(embeddings)

        # Phase 2.2: Compute graph metrics if available
        graph_metrics = {}
        if cooccurrence_graph and HAS_NETWORKX:
            import networkx as nx

            try:
                pagerank = nx.pagerank(cooccurrence_graph, weight="weight")
                betweenness = nx.betweenness_centrality(
                    cooccurrence_graph, weight="weight"
                )
                graph_metrics = {
                    "pagerank": pagerank,
                    "betweenness": betweenness,
                }
            except Exception as e:
                logger.warning(f"Graph metrics computation failed: {e}")

        # Group keywords by cluster
        clusters = {}
        for i, (keyword, label) in enumerate(zip(all_keywords, cluster_labels)):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append((keyword, i))

        # Create KeywordCluster objects with enhanced metadata
        keyword_clusters = []
        for label, keywords_with_idx in clusters.items():
            keywords_list = [kw for kw, _ in keywords_with_idx]

            # Find primary keyword (highest importance, graph centrality if available)
            primary_kw = keywords_list[0]
            if primary:
                for kw in primary:
                    if kw in keywords_list:
                        primary_kw = kw
                        break

            # Use graph centrality to refine primary if available
            if graph_metrics.get("pagerank"):
                kw_normalized = {
                    self._normalize_keyword(kw): kw for kw in keywords_list
                }
                best_centrality = -1
                for kw_norm, kw in kw_normalized.items():
                    centrality = graph_metrics["pagerank"].get(kw_norm, 0)
                    if centrality > best_centrality:
                        best_centrality = centrality
                        primary_kw = kw

            # Get topic theme if available
            topic_theme = primary_kw
            if topic_info is not None and topic_labels is not None:
                # Find most common topic in this cluster
                cluster_topics = [topic_labels[i] for kw, i in keywords_with_idx]
                if cluster_topics:
                    from collections import Counter

                    most_common_topic = Counter(cluster_topics).most_common(1)[0][0]
                    if most_common_topic >= 0:  # -1 is outlier topic
                        try:
                            topic_name = topic_info[
                                topic_info["Topic"] == most_common_topic
                            ]["Name"].values[0]
                            topic_theme = topic_name if topic_name else primary_kw
                        except:
                            pass

            # Phase 2.2: Store graph metrics in cluster metadata if available
            cluster_metadata = {}
            if graph_metrics.get("pagerank"):
                cluster_pagerank = (
                    sum(
                        graph_metrics["pagerank"].get(self._normalize_keyword(kw), 0)
                        for kw in keywords_list
                    )
                    / len(keywords_list)
                    if keywords_list
                    else 0
                )
                cluster_metadata["avg_pagerank"] = cluster_pagerank

            # Phase 4.2: Add competitiveness metadata if SERP data available
            # (Would need to pass SERP data through inputs/context)
            # For now, cluster metadata is stored but not exposed in KeywordCluster model

            keyword_clusters.append(
                KeywordCluster(
                    cluster_name=primary_kw,
                    keywords=keywords_list,
                    primary_keyword=primary_kw,
                    topic_theme=topic_theme,
                )
            )

        return keyword_clusters

    async def _get_seo_metrics(
        self,
        parsed_doc: Dict[str, Any],
        inputs: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Get SEO metrics (difficulty, search volume) - Phase 4 enhanced.

        Uses:
        - OpenPageRank API for domain authority
        - LLM-based SERP analysis for competition insights
        """
        from marketing_project.services.engines.seo_keywords.seo_metrics_provider import (
            SEOMetricsProvider,
        )

        keywords = inputs.get("keywords", [])
        if not keywords:
            primary = inputs.get("primary_keywords", [])
            secondary = inputs.get("secondary_keywords", [])
            long_tail = inputs.get("long_tail_keywords", [])
            keywords = primary + secondary + long_tail

        # Get pipeline from inputs (passed from composer)
        pipeline = inputs.get("_pipeline")
        job_id = context.get("job_id")

        # Initialize provider with pipeline
        provider = SEOMetricsProvider(pipeline=pipeline)

        # Analyze SERP for top keywords using LLM (Phase 4.1)
        serp_data = {}
        top_keywords = keywords[:10]  # Limit to top 10 for performance

        # Get SERP analysis model from engine config if available
        serp_model = None
        if pipeline and hasattr(pipeline, "pipeline_config"):
            step_config = pipeline.pipeline_config.get_step_config("seo_keywords")
            if step_config and hasattr(step_config, "seo_keywords_engine_config"):
                engine_config = step_config.seo_keywords_engine_config
                if engine_config and engine_config.serp_analysis_model:
                    serp_model = engine_config.serp_analysis_model

        for keyword in top_keywords:
            try:
                serp_analysis = await provider.analyze_serp_with_llm(
                    keyword, pipeline=pipeline, job_id=job_id, model=serp_model
                )
                serp_data[keyword] = serp_analysis
            except Exception as e:
                logger.warning(f"SERP analysis failed for '{keyword}': {e}")

        # Get difficulty scores (uses SERP data if available)
        difficulty = await provider.get_keyword_difficulty(keywords, serp_data)

        # Get metadata
        metadata = await provider.get_keyword_metadata(keywords, serp_data)

        # Calculate search volume summary
        primary = inputs.get("primary_keywords", [])
        secondary = inputs.get("secondary_keywords", [])
        long_tail = inputs.get("long_tail_keywords", [])

        primary_volume = sum(
            m.search_volume or 0 for m in metadata if m.keyword in primary
        )
        secondary_volume = sum(
            m.search_volume or 0 for m in metadata if m.keyword in secondary
        )
        long_tail_volume = sum(
            m.search_volume or 0 for m in metadata if m.keyword in long_tail
        )

        return {
            "keyword_difficulty": difficulty,
            "metadata": metadata,
            "search_volume_summary": {
                "primary": primary_volume,
                "secondary": secondary_volume,
                "long_tail": long_tail_volume,
            },
            "_serp_data": serp_data,  # Store for optimization recommendations
        }
