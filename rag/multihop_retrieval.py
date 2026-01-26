"""
Multi-hop Retrieval for RAG System (Practice 4)

Decomposes complex questions into sub-queries for better retrieval
on questions requiring information from multiple document sections.

Features:
- Query decomposition using LLM
- Sequential sub-query retrieval
- Context aggregation
- Optional enable/disable
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Any

from langchain_core.documents import Document


@dataclass
class MultiHopConfig:
    """Configuration for multi-hop retrieval."""
    enabled: bool = False
    max_hops: int = 3  # Maximum sub-queries
    min_complexity_words: int = 10  # Minimum words to trigger decomposition
    aggregate_contexts: bool = True  # Combine contexts from all hops


@dataclass
class SubQuery:
    """A decomposed sub-query."""
    query: str
    intent: str  # What this sub-query is trying to find
    order: int  # Execution order


@dataclass
class HopResult:
    """Result from a single retrieval hop."""
    sub_query: SubQuery
    documents: List[Document]
    scores: Optional[List[float]] = None


@dataclass
class MultiHopResult:
    """Result from multi-hop retrieval."""
    original_query: str
    sub_queries: List[SubQuery]
    hop_results: List[HopResult]
    aggregated_documents: List[Document]
    was_decomposed: bool


class QueryDecomposer:
    """
    Decomposes complex queries into simpler sub-queries.

    Uses LLM to analyze query complexity and generate focused sub-queries.
    """

    # Keywords suggesting multi-part questions
    COMPLEXITY_INDICATORS = [
        "and", "also", "both", "compare", "difference between",
        "relationship between", "how does", "what are the",
        "multiple", "several", "various", "as well as",
        "in addition", "furthermore", "moreover",
    ]

    def __init__(self, llm_fn: Optional[Callable[[str], str]] = None):
        """
        Initialize query decomposer.

        Args:
            llm_fn: Function that takes a prompt and returns LLM response.
                    If None, uses rule-based decomposition.
        """
        self.llm_fn = llm_fn

    def is_complex(self, query: str, config: MultiHopConfig) -> bool:
        """
        Determine if a query is complex enough to benefit from decomposition.

        Args:
            query: The query to analyze
            config: Multi-hop configuration

        Returns:
            True if query should be decomposed
        """
        # Check word count
        words = query.split()
        if len(words) < config.min_complexity_words:
            return False

        # Check for complexity indicators
        query_lower = query.lower()
        for indicator in self.COMPLEXITY_INDICATORS:
            if indicator in query_lower:
                return True

        # Check for multiple question marks
        if query.count("?") > 1:
            return True

        # Check for enumeration patterns
        if re.search(r"\d+\.\s|\d+\)", query):
            return True

        return False

    def decompose(self, query: str, config: MultiHopConfig) -> List[SubQuery]:
        """
        Decompose a complex query into sub-queries.

        Args:
            query: The complex query
            config: Multi-hop configuration

        Returns:
            List of SubQuery objects
        """
        if self.llm_fn:
            return self._decompose_with_llm(query, config)
        return self._decompose_rule_based(query, config)

    def _decompose_with_llm(self, query: str, config: MultiHopConfig) -> List[SubQuery]:
        """Use LLM to decompose query."""
        prompt = f"""Analyze this question and break it into simpler sub-questions that can be answered independently.
Return at most {config.max_hops} sub-questions.

Question: {query}

Format your response as a numbered list:
1. [First sub-question] - Intent: [what information this seeks]
2. [Second sub-question] - Intent: [what information this seeks]
...

If the question is simple and doesn't need decomposition, respond with:
SIMPLE: [original question]
"""
        try:
            response = self.llm_fn(prompt)
            return self._parse_llm_response(response, query)
        except Exception as e:
            print(f"LLM decomposition failed: {e}, falling back to rule-based")
            return self._decompose_rule_based(query, config)

    def _parse_llm_response(self, response: str, original_query: str) -> List[SubQuery]:
        """Parse LLM response into SubQuery objects."""
        if response.strip().startswith("SIMPLE:"):
            return [SubQuery(query=original_query, intent="original", order=0)]

        sub_queries = []
        lines = response.strip().split("\n")

        for i, line in enumerate(lines):
            # Match patterns like "1. question - Intent: intent" or "1) question"
            match = re.match(r"^\d+[\.\)]\s*(.+?)(?:\s*-\s*Intent:\s*(.+))?$", line.strip())
            if match:
                query_text = match.group(1).strip()
                intent = match.group(2).strip() if match.group(2) else "sub-query"
                sub_queries.append(SubQuery(
                    query=query_text,
                    intent=intent,
                    order=i,
                ))

        if not sub_queries:
            return [SubQuery(query=original_query, intent="original", order=0)]

        return sub_queries[:3]  # Max 3 sub-queries

    def _decompose_rule_based(self, query: str, config: MultiHopConfig) -> List[SubQuery]:
        """Rule-based query decomposition."""
        sub_queries = []

        # Split on common conjunctions
        parts = re.split(r"\s+and\s+|\s+also\s+|,\s*and\s+|\.\s+", query, flags=re.IGNORECASE)
        parts = [p.strip() for p in parts if p.strip() and len(p.strip()) > 10]

        if len(parts) > 1:
            for i, part in enumerate(parts[:config.max_hops]):
                # Ensure it's a proper question
                if not part.endswith("?"):
                    part = part.rstrip(".") + "?"
                sub_queries.append(SubQuery(
                    query=part,
                    intent=f"part {i+1}",
                    order=i,
                ))
        else:
            # No decomposition needed
            sub_queries.append(SubQuery(
                query=query,
                intent="original",
                order=0,
            ))

        return sub_queries


class MultiHopRetriever:
    """
    Multi-hop retriever that decomposes queries and aggregates results.
    """

    def __init__(
        self,
        retrieve_fn: Callable[[str, int], List[Document]],
        decomposer: Optional[QueryDecomposer] = None,
        config: Optional[MultiHopConfig] = None,
    ):
        """
        Initialize multi-hop retriever.

        Args:
            retrieve_fn: Function that retrieves documents for a query
            decomposer: QueryDecomposer instance
            config: Multi-hop configuration
        """
        self.retrieve_fn = retrieve_fn
        self.decomposer = decomposer or QueryDecomposer()
        self.config = config or MultiHopConfig()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        config: Optional[MultiHopConfig] = None,
    ) -> MultiHopResult:
        """
        Perform multi-hop retrieval.

        Args:
            query: The user query
            top_k: Documents per sub-query
            config: Override configuration

        Returns:
            MultiHopResult with all retrieval data
        """
        cfg = config or self.config

        if not cfg.enabled:
            # Direct retrieval without decomposition
            docs = self.retrieve_fn(query, top_k)
            return MultiHopResult(
                original_query=query,
                sub_queries=[SubQuery(query=query, intent="direct", order=0)],
                hop_results=[HopResult(
                    sub_query=SubQuery(query=query, intent="direct", order=0),
                    documents=docs,
                )],
                aggregated_documents=docs,
                was_decomposed=False,
            )

        # Check complexity
        if not self.decomposer.is_complex(query, cfg):
            docs = self.retrieve_fn(query, top_k)
            return MultiHopResult(
                original_query=query,
                sub_queries=[SubQuery(query=query, intent="simple", order=0)],
                hop_results=[HopResult(
                    sub_query=SubQuery(query=query, intent="simple", order=0),
                    documents=docs,
                )],
                aggregated_documents=docs,
                was_decomposed=False,
            )

        # Decompose query
        sub_queries = self.decomposer.decompose(query, cfg)

        # Execute sub-queries
        hop_results = []
        for sq in sub_queries:
            docs = self.retrieve_fn(sq.query, top_k)
            hop_results.append(HopResult(
                sub_query=sq,
                documents=docs,
            ))

        # Aggregate documents
        if cfg.aggregate_contexts:
            aggregated = self._aggregate_documents(hop_results, top_k)
        else:
            # Just use first hop results
            aggregated = hop_results[0].documents if hop_results else []

        return MultiHopResult(
            original_query=query,
            sub_queries=sub_queries,
            hop_results=hop_results,
            aggregated_documents=aggregated,
            was_decomposed=len(sub_queries) > 1,
        )

    def _aggregate_documents(
        self,
        hop_results: List[HopResult],
        top_k: int,
    ) -> List[Document]:
        """
        Aggregate documents from multiple hops, removing duplicates.

        Uses round-robin selection to ensure diversity.
        """
        seen_content = set()
        aggregated = []

        # Round-robin through hop results
        max_len = max(len(hr.documents) for hr in hop_results) if hop_results else 0

        for i in range(max_len):
            for hr in hop_results:
                if i < len(hr.documents):
                    doc = hr.documents[i]
                    content_hash = hash(doc.page_content[:200])

                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        aggregated.append(doc)

                        if len(aggregated) >= top_k * 2:  # Allow more for multi-hop
                            return aggregated[:top_k * 2]

        return aggregated


def create_multihop_retriever(
    retrieve_fn: Callable[[str, int], List[Document]],
    llm_fn: Optional[Callable[[str], str]] = None,
    enabled: bool = False,
    max_hops: int = 3,
) -> MultiHopRetriever:
    """
    Factory function to create a multi-hop retriever.

    Args:
        retrieve_fn: Base retrieval function
        llm_fn: Optional LLM function for intelligent decomposition
        enabled: Whether multi-hop is enabled
        max_hops: Maximum number of sub-queries

    Returns:
        Configured MultiHopRetriever instance
    """
    decomposer = QueryDecomposer(llm_fn=llm_fn)
    config = MultiHopConfig(enabled=enabled, max_hops=max_hops)
    return MultiHopRetriever(
        retrieve_fn=retrieve_fn,
        decomposer=decomposer,
        config=config,
    )
