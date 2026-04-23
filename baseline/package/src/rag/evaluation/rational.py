class ExplainableOrchestrator:
    """
    Enhanced orchestrator that provides detailed rationales for decisions.
    """

    def __init__(self, bm25, dense, graph_rag):
        self.bm25 = bm25
        self.dense = dense
        self.graph_rag = graph_rag

    def analyze_query_features(self, query: str) -> dict:
        """Extract features from query to guide orchestration."""
        q_lower = query.lower()
        tokens = query.split()

        features = {
            "length": len(tokens),
            "has_digits": any(ch.isdigit() for ch in query),
            "has_quotes": ('"' in query) or ("'" in query),
            "has_names": any(t[0].isupper() for t in tokens if len(t) > 1),
            "question_words": sum(
                1
                for w in ["who", "what", "when", "where", "why", "how"]
                if w in q_lower
            ),
            "complexity": "simple" if len(tokens) <= 6 else "complex",
        }

        # Classify query type
        features["is_factoid"] = (
            features["has_digits"] or features["has_quotes"] or features["length"] <= 6
        )
        features["is_semantic"] = features["length"] > 10 and not features["has_digits"]

        return features

    def explainable_route(self, query: str, top_k: int = 5) -> tuple[list, dict]:
        """Route query with detailed explanation."""
        features = self.analyze_query_features(query)
        explanation = {
            "query": query,
            "features": features,
            "decisions": [],
            "weights": {},
            "rationale": "",
        }

        # Determine weights based on features
        if features["is_factoid"]:
            weights = {"bm25": 1.4, "dense": 0.9, "graph": 0.5}
            rationale = f"Query is FACTOID (digits={features['has_digits']}, quotes={features['has_quotes']}, length={features['length']}). "
            rationale += "BM25 weighted higher (1.4) for exact matching. Dense reduced (0.9). Graph minimal (0.5)."
            explanation["decisions"].append(
                "Route: Factoid query detected → BM25-heavy strategy"
            )
        elif features["is_semantic"]:
            weights = {"bm25": 1.0, "dense": 1.3, "graph": 0.8}
            rationale = f"Query is SEMANTIC (length={features['length']}, no digits). "
            rationale += "Dense weighted higher (1.3) for semantic similarity. BM25 standard (1.0). Graph elevated (0.8) for context."
            explanation["decisions"].append(
                "Route: Semantic query detected → Dense-heavy strategy"
            )
        else:
            weights = {"bm25": 1.2, "dense": 1.0, "graph": 0.6}
            rationale = "Query is BALANCED. Using default weights: BM25 (1.2), Dense (1.0), Graph (0.6)."
            explanation["decisions"].append("Route: Balanced query → Standard fusion")

        explanation["weights"] = weights
        explanation["rationale"] = rationale

        # Retrieve from all agents
        pre_k = max(30, top_k * 10)
        explanation["decisions"].append(f"Retrieving top-{pre_k} from each agent")

        bm25_docs = self.bm25.search(query, top_k=pre_k)
        dense_docs = self.dense.search(query, top_k=pre_k)
        graph_docs = self.graph_rag.retrieve(query, top_k=pre_k)

        explanation["decisions"].append(
            f"BM25: {len(bm25_docs)} docs, Dense: {len(dense_docs)} docs, Graph: {len(graph_docs)} docs"
        )

        # Simple fusion (placeholder - should use proper RRF fusion)
        # Note: This is a simplified version. In production, use proper RRF fusion
        from collections import defaultdict

        scores = defaultdict(float)
        doc_store = {}
        k_rrf = 60

        for name, docs, weight in [
            ("bm25", bm25_docs, weights["bm25"]),
            ("dense", dense_docs, weights["dense"]),
            ("graph", graph_docs, weights["graph"]),
        ]:
            for rank, d in enumerate(docs, 1):
                uid = d.metadata.get("chunk_id") or d.metadata.get("record_id")
                if uid:
                    doc_store[uid] = d
                    scores[uid] += weight * (1.0 / (k_rrf + rank))

        fused = sorted(
            doc_store.values(),
            key=lambda d: scores[
                d.metadata.get("chunk_id") or d.metadata.get("record_id")
            ],
            reverse=True,
        )
        explanation["decisions"].append(
            f"Fusion complete: {len(fused)} unique documents"
        )

        final_docs = fused[:top_k]
        explanation["decisions"].append(f"Returning top-{top_k} documents")

        return final_docs, explanation

    def print_explanation(self, explanation: dict):
        """Print explanation in readable format."""
        print("\n" + "=" * 80)
        print(f"QUERY: {explanation['query']}")
        print("=" * 80)
        print("\nQUERY FEATURES:")
        for k, v in explanation["features"].items():
            print(f"  {k}: {v}")
        print("\nAGENT WEIGHTS:")
        for k, v in explanation["weights"].items():
            print(f"  {k}: {v}")
        print("\nRATIONALE:")
        print(f"  {explanation['rationale']}")
        print("\nDECISION FLOW:")
        for i, dec in enumerate(explanation["decisions"], 1):
            print(f"  {i}. {dec}")
        print("=" * 80)
