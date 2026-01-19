# tests/test_planner.py
import asyncio
import json
import pytest
from unittest.mock import AsyncMock

from agentforge.core.planner import (
    AdvancedPlanner,
    PlanGraph,
    PlanNode,
    ScoreRationale
)


# ---------------------------
# Mock LLM for deterministic behavior
# ---------------------------

class MockLLM:
    def __init__(self, responses=None):
        self.responses = responses or {}
        self.call_count = 0

    async def generate(self, prompt: str, **kwargs) -> str:
        self.call_count += 1
        # Basit eşleşme: prompt içindeki anahtar kelimelere göre yanıt döndür
        if "candidate" in prompt.lower():
            return json.dumps([
                {"type": "tool", "name": "search_web", "input": {"query": "AI agents"}},
                {"type": "final", "output": "No need to search."},
                {"type": "tool", "name": "read_file", "input": {"path": "report.txt"}}
            ])
        elif "score" in prompt.lower():
            return json.dumps({"score": 0.85, "rationale": "Good candidate"})
        elif "refine" in prompt.lower():
            return json.dumps({"type": "tool", "name": "search_web", "input": {"query": "updated query"}})
        elif "fallback" in prompt.lower() or "single-step" in prompt.lower():
            return json.dumps({"type": "tool", "name": "default_tool", "input": {}})
        elif "decompose" in prompt.lower():
            return json.dumps([
                {"agent": "Researcher", "task": "Find latest papers"},
                {"agent": "Coder", "task": "Implement agent"}
            ])
        else:
            return "[MOCK RESPONSE]"

    async def generate_with_functions(self, prompt: str, functions, **kwargs):
        # Simulate OpenAI-style function call
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "function_call": {
                        "name": "search_web",
                        "arguments": json.dumps({"query": "AI agent frameworks"})
                    }
                },
                "finish_reason": "function_call"
            }]
        }

    async def stream(self, prompt: str, **kwargs):
        yield "[MOCK STREAM]"

class TestPlanGraph:
    def test_add_node_and_navigation(self):
        graph = PlanGraph()
        root = graph.add_node({"raw": "root"}, {"type": "tool", "name": "A"})
        child = graph.add_node({"raw": "child"}, {"type": "tool", "name": "B"}, parent_id=root.id)

        assert len(graph.nodes) == 2
        assert graph.children_of(root.id) == [child]
        assert graph.all_roots() == [root]
        path = graph.path_to_root(child.id)
        assert [n.id for n in path] == [root.id, child.id]

class TestCandidateGeneration:
    @pytest.mark.asyncio
    async def test_generate_candidates_valid_json(self):
        llm = MockLLM()
        planner = AdvancedPlanner(llm)
        candidates = await planner._generate_candidates("Generate candidates", n=3)
        assert len(candidates) == 3
        assert candidates[0]["type"] == "tool"
        assert candidates[0]["name"] == "search_web"

    @pytest.mark.asyncio
    async def test_generate_candidates_invalid_json_fallback(self):
        llm = MockLLM(responses={"candidate": "Invalid JSON { not valid }"})
        planner = AdvancedPlanner(llm)
        candidates = await planner._generate_candidates("Generate candidates", n=3)
        # Should retry with stricter prompt
        assert len(candidates) == 0  # veya fallback davranışı

    @pytest.mark.asyncio
    async def test_normalize_candidate_variants(self):
        planner = AdvancedPlanner(MockLLM())

        # Test various input shapes
        assert planner._normalize_candidate({"type": "tool", "name": "x", "input": {"a": 1}}) == \
               {"type": "tool", "name": "x", "input": {"a": 1}}

        assert planner._normalize_candidate({"action": "tool", "tool": "y", "args": {"b": 2}}) == \
               {"type": "tool", "name": "y", "input": {"b": 2}}

        assert planner._normalize_candidate({"name": "z", "arguments": '{"c": 3}'}) == \
               {"type": "tool", "name": "z", "input": {"c": 3}}

class TestScoringAndSelection:
    @pytest.mark.asyncio
    async def test_score_candidate(self):
        llm = MockLLM()
        planner = AdvancedPlanner(llm)
        score, rationale = await planner._score_candidate(
            task="test task",
            history=[],
            context={},
            candidate={"type": "tool", "name": "test"}
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert "Good candidate" in rationale

    @pytest.mark.asyncio
    async def test_select_best_node(self):
        graph = PlanGraph()
        node1 = graph.add_node({}, {"type": "tool", "name": "A"})
        node1.score = 0.9
        node2 = graph.add_node({}, {"type": "tool", "name": "B"})
        node2.score = 0.85

        llm = MockLLM()
        planner = AdvancedPlanner(llm)
        best = await planner._select_best_node(graph, "task", [], {})
        assert best.id == node1.id

class TestDeadEndDetection:
    def test_is_dead_end_repeated_tool(self):
        graph = PlanGraph()
        node1 = graph.add_node({}, {"type": "tool", "name": "search", "input": {"q": "AI"}})
        node2 = graph.add_node({}, {"type": "tool", "name": "search", "input": {"q": "AI"}}, parent_id=node1.id)

        planner = AdvancedPlanner(MockLLM())
        assert planner._is_dead_end(graph, node2) is True

    def test_is_dead_end_different_inputs(self):
        graph = PlanGraph()
        node1 = graph.add_node({}, {"type": "tool", "name": "search", "input": {"q": "AI"}})
        node2 = graph.add_node({}, {"type": "tool", "name": "search", "input": {"q": "LLM"}})

        planner = AdvancedPlanner(MockLLM())
        assert planner._is_dead_end(graph, node2) is False

class TestSingleStepFallback:
    @pytest.mark.asyncio
    async def test_fallback_with_function_calling(self):
        llm = MockLLM()
        planner = AdvancedPlanner(llm)
        tools = {"search_web": {"name": "search_web", "parameters": {}}}

        result = await planner._single_step_fallback("Search for AI agents", tools=tools)
        assert result["type"] == "tool"
        assert result["name"] == "search_web"
        assert "AI agent frameworks" in json.dumps(result["input"])

  class TestContextBuilding:
    def test_build_base_prompt_includes_peer_messages(self):
        planner = AdvancedPlanner(MockLLM())
        context = {
            "semantic": [{"text": "Semantic fact", "score": 0.9}],
            "episodic": [{"action": "search", "result": "found X"}],
            "peer_messages": "[Researcher] (info): Found key paper on agents"
        }
        prompt = planner._build_base_prompt("task", [], context)

        assert "PEER CONTEXT" in prompt
        assert "Found key paper" in prompt
        assert "Semantic fact" in prompt
        assert "found X" in prompt

 @pytest.mark.asyncio
async def test_planner_full_flow():
    llm = MockLLM()
    planner = AdvancedPlanner(llm, n_candidates=2, max_depth=2)

    tools = {
        "search_web": {
            "name": "search_web",
            "description": "Search the web",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}
        }
    }

    result = await planner.plan(
        task="Research AI agents and summarize",
        history=[],
        context={"semantic": [], "episodic": [], "peer_messages": "(no peer messages)"},
        tools=tools
    )

    # Should return first actionable step from best path
    assert "type" in result
    assert result["type"] in ("tool", "final")

class TestErrorResilience:
    @pytest.mark.asyncio
    async def test_plan_handles_llm_failure_gracefully(self):
        class FailingLLM:
            async def generate(self, *args, **kwargs):
                raise RuntimeError("LLM failed")

            async def generate_with_functions(self, *args, **kwargs):
                raise NotImplementedError()

        planner = AdvancedPlanner(FailingLLM())
        result = await planner.plan("simple task", [], {})

        # Should fall back to single-step, which also fails → final fallback
        assert result["type"] == "final"
        assert "failed" in result["output"]

  
