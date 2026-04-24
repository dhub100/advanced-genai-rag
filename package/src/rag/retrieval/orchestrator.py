from rag.retrieval.agents.answer_synthesizer import AnswerSynthesizerAgent
from rag.retrieval.retriever.confidence import ConfidenceRetriever
from rag.retrieval.retriever.voting import VotingRetriever
from rag.retrieval.retriever.waterfall import WaterfallRetriever


class Orchestrator(WaterfallRetriever, ConfidenceRetriever, VotingRetriever):
    """Main orchestrator combining retrieval strategies and answer synthesis."""

    def __init__(self, bm25, dense, graph, synthesizer: AnswerSynthesizerAgent):
        super().__init__(bm25, dense, graph)
        self.bm25 = bm25
        self.dense = dense
        self.graph = graph
        self.synthesizer = synthesizer

    def run(self, strategy: str, query: str, top_k=5):
        if strategy == "waterfall":
            docs, trace = self.waterfall_orchestrate(query, top_k)
        elif strategy == "voting":
            docs, trace = self.voting_orchestrate(query, top_k)
        elif strategy == "confidence":
            docs, trace = self.confidence_orchestrate(query, top_k)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        answer = self.synthesizer.generate(query, docs)

        return {
            "query": query,
            "strategy": strategy,
            "trace": trace,
            "documents": docs,
            "answer": answer,
        }
