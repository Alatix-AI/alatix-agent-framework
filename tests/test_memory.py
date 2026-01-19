# tests/test_memory.py
import asyncio
import os
import tempfile
import shutil
from pathlib import Path
import pytest
import numpy as np
from unittest.mock import AsyncMock

from core.memory import (
    SimpleVectorStore,
    FAISSVectorStore,
    EpisodicMemoryAdvanced,
    SemanticMemoryAdvanced,
    MemorySystem
)


# ---------------------------
# Mock LLM for embedding
# ---------------------------

class MockLLM:
    async def embed(self, texts):
        # Deterministic mock embedding: her metin için sabit vektör
        if isinstance(texts, str):
            texts = [texts]
        return [[float(ord(c)) for c in text[:8]] + [0.0] * (8 - min(8, len(text))) for text in texts]


# ---------------------------
# Vector Store Tests
# ---------------------------

class TestSimpleVectorStore:
    def test_add_and_search(self):
        store = SimpleVectorStore(dim=3)
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        meta1 = {"id": 1, "text": "first"}
        meta2 = {"id": 2, "text": "second"}

        store.add(vec1, meta1)
        store.add(vec2, meta2)

        query = np.array([1.0, 0.1, 0.0])
        results = store.search_by_vector(query, k=1)
        assert len(results) == 1
        score, meta = results[0]
        assert meta["id"] == 1
        assert score > 0.9  # cosine similarity

    def test_dimension_mismatch_raises_error(self):
        store = SimpleVectorStore(dim=2)
        with pytest.raises(ValueError, match="Vector dim mismatch"):
            store.add(np.array([1.0, 0.0, 0.0]), {"id": 1})


class TestFAISSVectorStore:
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_add_and_search(self):
        index_path = os.path.join(self.temp_dir, "test.index")
        store = FAISSVectorStore(dim=3, index_path=index_path)
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        meta1 = {"id": 1}
        meta2 = {"id": 2}

        store.add(vec1, meta1)
        store.add(vec2, meta2)

        query = np.array([1.0, 0.1, 0.0])
        results = store.search_by_vector(query, k=1)
        assert len(results) == 1
        score, meta = results[0]
        assert meta["id"] == 1

    def test_persistence(self):
        index_path = os.path.join(self.temp_dir, "persist.index")
        store1 = FAISSVectorStore(dim=2, index_path=index_path)
        store1.add(np.array([1.0, 0.0]), {"text": "hello"})
        # Simulate save
        store1._save_to_disk()

        # Load in new instance
        store2 = FAISSVectorStore(dim=2, index_path=index_path)
        assert len(store2) == 1
        assert store2.metadata[0]["text"] == "hello"


# ---------------------------
# Episodic Memory Tests
# ---------------------------

class TestEpisodicMemoryAdvanced:
    @pytest.mark.asyncio
    async def test_add_and_retrieve(self):
        llm = MockLLM()
        memory = EpisodicMemoryAdvanced(llm, max_items=10, chunk_size=3)
        await memory.add({"action": "search", "result": "found X"})
        await memory.add({"action": "analyze", "result": "X is valid"})

        recent = memory.search_last(k=2)
        assert len(recent) == 2
        assert recent[0]["action"] == "analyze"

    @pytest.mark.asyncio
    async def test_summarization_triggered(self):
        llm = MockLLM()
        memory = EpisodicMemoryAdvanced(llm, max_items=10, chunk_size=2)
        for i in range(4):  # 2 chunks
            await memory.add({"action": f"step_{i}", "result": f"result_{i}"})

        # Wait for background task
        await asyncio.sleep(0.1)

        summaries = memory.get_recent_summaries(n=2)
        assert len(summaries) >= 1
        assert "summary" in summaries[0]

    @pytest.mark.asyncio
    async def test_persistent_save_load(self):
        temp_dir = tempfile.mkdtemp()
        try:
            path = Path(temp_dir) / "episodic.pkl"
            llm = MockLLM()
            memory1 = EpisodicMemoryAdvanced(llm, persistent=True, episodic_path=path)
            await memory1.add({"action": "test", "result": "value"})
            # Force save
            await memory1._save_to_disk()

            memory2 = EpisodicMemoryAdvanced(llm, persistent=True, episodic_path=path)
            assert len(memory2.get_all()) == 1
            assert memory2.get_all()[0]["action"] == "test"
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


# ---------------------------
# Semantic Memory Tests
# ---------------------------

class TestSemanticMemoryAdvanced:
    @pytest.mark.asyncio
    async def test_add_and_search(self):
        llm = MockLLM()
        memory = SemanticMemoryAdvanced(llm, dim=8, max_items=10)
        await memory.add("The sky is blue", {"source": "observation"})

        results = await memory.search("What color is the sky?", k=1)
        assert len(results) == 1
        assert "sky" in results[0]["text"]
        assert results[0]["source"] == "observation"

    @pytest.mark.asyncio
    async def test_importance_decay_and_pruning(self):
        llm = MockLLM()
        memory = SemanticMemoryAdvanced(
            llm,
            dim=8,
            max_items=2,
            forget_threshold=0.5,
            importance_decay=0.5
        )
        # Add low-importance item
        await memory.add("low value", {"importance": 0.1})
        # Add high-importance items
        await memory.add("high1", {"importance": 2.0})
        await memory.add("high2", {"importance": 2.0})

        # Should prune low-importance
        results = await memory.search("query", k=3)
        texts = [r["text"] for r in results]
        assert "low value" not in texts
        assert len(texts) == 2

    @pytest.mark.asyncio
    async def test_embedding_dimension_mismatch(self):
        llm = MockLLM()  # returns 8-dim
        memory = SemanticMemoryAdvanced(llm, dim=4)  # expects 4-dim
        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            await memory.add("test text")


# ---------------------------
# Memory System Integration Tests
# ---------------------------

class TestMemorySystem:
    @pytest.mark.asyncio
    async def test_add_episode_and_knowledge(self):
        llm = MockLLM()
        memory = MemorySystem(llm, episodic_limit=5, embed_dim=8)

        # Add episode
        await memory.add_episode("tool_call", {"status": "success"})

        # Add knowledge
        await memory.add_knowledge("Learned fact", {"source": "tool"})

        # Retrieve context
        context = await memory.retrieve_context("query", k_semantic=1, k_episodic=1)
        assert "semantic" in context
        assert "episodic" in context
        assert len(context["semantic"]) == 1
        assert len(context["episodic"]) == 1

    @pytest.mark.asyncio
    async def test_persistent_memory_system(self):
        temp_dir = tempfile.mkdtemp()
        try:
            llm = MockLLM()
            name = "test_agent"
            path_base = os.path.join(temp_dir, name)

            # First instance
            mem1 = MemorySystem(llm, persistent=True, name=name)
            await mem1.add_knowledge("Persistent fact", {"task": "test"})
            await mem1.add_episode("action", "result")

            # Second instance (should load)
            mem2 = MemorySystem(llm, persistent=True, name=name)
            context = await mem2.retrieve_context("fact", k_semantic=1, k_episodic=1)
            assert len(context["semantic"]) == 1
            assert "Persistent fact" in context["semantic"][0]["text"]
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


# ---------------------------
# Async Safety Test (Optional but good)
# ---------------------------

@pytest.mark.asyncio
async def test_concurrent_access_safety():
    llm = MockLLM()
    memory = EpisodicMemoryAdvanced(llm, max_items=100)

    async def add_items(start_id):
        for i in range(10):
            await memory.add({"id": start_id + i, "val": f"item_{start_id}_{i}"})

    # Run concurrently
    await asyncio.gather(
        add_items(0),
        add_items(100),
        add_items(200)
    )

    all_items = memory.get_all()
    assert len(all_items) == 30
    ids = {item["id"] for item in all_items}
    assert len(ids) == 30  # no duplicates → thread-safe
