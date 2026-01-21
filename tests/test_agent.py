# tests/test_agent.py
import asyncio
import json
import pytest
from unittest.mock import AsyncMock

from core.agent import Agent
from core.tools import Tool
from core.memory import MemorySystem
from core.llm import MockAdapter


# ---------------------------
# Mock LLM for deterministic behavior
# ---------------------------

class MockLLM:
    def __init__(self, responses=None):
        self.responses = responses or {}
        self.call_count = 0

    async def generate(self, prompt: str, **kwargs) -> str:
        self.call_count += 1
        if "tool" in prompt.lower():
            return json.dumps({"type": "tool", "name": "mock_tool", "input": {"query": "test"}})
        else:
            return json.dumps({"type": "final", "output": "Final answer from mock LLM"})

    async def stream(self, prompt: str, **kwargs):
        yield "Final "
        yield "answer "
        yield "streamed."

    async def embed(self, texts):
        # Deterministic mock embedding
        return [[float(ord(c)) for c in (text[:8] + " " * (8 - min(8, len(text))))] for text in texts]

    async def generate_with_functions(self, prompt, functions, **kwargs):
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "function_call": {
                        "name": "mock_tool",
                        "arguments": json.dumps({"query": "test"})
                    }
                },
                "finish_reason": "function_call"
            }]
        }


# ---------------------------
# Mock Tool
# ---------------------------

def mock_tool(query: str) -> str:
    return f"Result for {query}"


# ---------------------------
# Test Fixtures
# ---------------------------

@pytest.fixture
def mock_llm():
    return MockLLM()


@pytest.fixture
def mock_tool_obj():
    return Tool("mock_tool", mock_tool, description="A mock tool for testing")


@pytest.fixture
def agent(mock_tool_obj):
    # Create agent with mock LLM and tool
    agent = Agent(
        model="mock:test",
        tools=[mock_tool_obj],
        max_steps=3,
        temperature=0.0,
        persistent=False
    )
    # Inject mock LLM
    agent.llm.client.adapter.llm = MockAdapter()
    return agent


# ---------------------------
# Core Tests
# ---------------------------

class TestAgentBasicFlow:
    @pytest.mark.asyncio
    async def test_run_returns_final_output(self, agent):
        result = await agent.run("What is the answer?")
        assert "Final answer" in result

    @pytest.mark.asyncio
    async def test_run_stream_yields_events(self, agent):
        events = []
        async for event in agent.run_stream("What is the answer?"):
            events.append(event)

        assert len(events) > 0
        assert any(e["type"] == "final" for e in events)


class TestToolExecution:
    @pytest.mark.asyncio
    async def test_tool_execution_and_memory(self, agent):
        # Force planner to return tool call
        original_plan = agent.planner.plan
        async def mock_plan(*args, **kwargs):
            return {"type": "tool", "name": "mock_tool", "input": {"query": "test"}}

        agent.planner.plan = mock_plan

        events = []
        async for event in agent.run_stream("Use tool"):
            events.append(event)

        # Should have tool event
        tool_events = [e for e in events if e["type"] == "tool"]
        assert len(tool_events) == 1
        assert tool_events[0]["name"] == "mock_tool"
        assert tool_events[0]["result"]["success"] is True
        assert "Result for test" in str(tool_events[0]["result"]["output"])

        # Check memory
        episodic = agent.memory.episodic.get_all()
        assert len(episodic) == 1
        assert episodic[0]["action"] == "mock_tool"

        semantic = await agent.memory.semantic.search("test", k=1)
        assert len(semantic) == 1
        assert "Result for test" in semantic[0]["text"]


class TestInboxSystem:
    @pytest.mark.asyncio
    async def test_inbox_message_integration(self, agent):
        # Send message before run
        agent.receive_message("PeerAgent", "Important context: use parameter X", priority="info")

        # Mock planner to see peer context
        original_build_prompt = agent.planner._build_base_prompt
        planner_saw_peer_context = False

        def mock_build_prompt(task, history, context):
            nonlocal planner_saw_peer_context
            if "Important context" in context.get("peer_messages", ""):
                planner_saw_peer_context = True
            return original_build_prompt(task, history, context)

        agent.planner._build_base_prompt = mock_build_prompt

        # Mock planner to return final immediately
        async def mock_plan(*args, **kwargs):
            return {"type": "final", "output": "Used peer context"}

        agent.planner.plan = mock_plan

        result = await agent.run("Do task")
        assert planner_saw_peer_context
        assert "Used peer context" in result

        # Check that peer message was saved to semantic memory
        semantic = await agent.memory.semantic.search("peer context", k=1)
        assert len(semantic) > 0
        assert "Important context" in semantic[0]["text"]
        assert semantic[0]["source"] == "peer_message"


class TestMemoryIntegration:
    @pytest.mark.asyncio
    async def test_episodic_and_semantic_memory_persistence(self, agent):
        # Mock tool execution
        async def mock_plan(*args, **kwargs):
            return {"type": "tool", "name": "mock_tool", "input": {"query": "memory_test"}}

        agent.planner.plan = mock_plan

        await agent.run("Test memory")

        # Check episodic
        episodic = agent.memory.episodic.search_last(k=1)
        assert len(episodic) == 1
        assert episodic[0]["action"] == "mock_tool"

        # Check semantic
        semantic = await agent.memory.semantic.search("memory_test", k=1)
        assert len(semantic) == 1
        assert "memory_test" in semantic[0]["text"]


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_agent_handles_tool_failure_gracefully(self, agent):
        def failing_tool():
            raise RuntimeError("Tool failed")

        failing_tool_obj = Tool("failing_tool", failing_tool)
        agent.tool_registry = type(agent.tool_registry)()
        agent.tool_registry.add(failing_tool_obj)

        async def mock_plan(*args, **kwargs):
            return {"type": "tool", "name": "failing_tool", "input": {}}

        agent.planner.plan = mock_plan

        events = []
        async for event in agent.run_stream("Run failing tool"):
            events.append(event)

        # Should not crash
        tool_events = [e for e in events if e["type"] == "tool"]
        assert len(tool_events) == 1
        assert tool_events[0]["result"]["success"] is False
        assert "Tool failed" in tool_events[0]["result"]["output"]["error"]


class TestMaxStepsLimit:
    @pytest.mark.asyncio
    async def test_agent_respects_max_steps(self, agent):
        agent.max_steps = 2

        # Mock planner to always return tool (infinite loop scenario)
        call_count = 0
        async def mock_plan(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return {"type": "tool", "name": "mock_tool", "input": {"query": f"step{call_count}"}}

        agent.planner.plan = mock_plan

        events = []
        async for event in agent.run_stream("Infinite task"):
            events.append(event)

        # Should stop after max_steps
        assert call_count == 2
        assert any("max_steps" in str(e) for e in events if e["type"] == "final")


# ---------------------------
# Integration Test: Full Flow
# ---------------------------

@pytest.mark.asyncio
async def test_agent_full_integration_flow():
    llm = MockLLM()
    tool_obj = Tool("search", lambda q: f"Results for {q}")

    # Create memory system explicitly
    memory = MemorySystem(llm, persistent=False)

    agent = Agent(
        model="mock:test",
        tools=[tool_obj],
        memory=memory,
        max_steps=2,
        temperature=0.0
    )
    agent.llm.client.adapter.llm = llm

    # Mock planner sequence: tool then final
    plan_sequence = [
        {"type": "tool", "name": "search", "input": {"query": "AI agents"}},
        {"type": "final", "output": "Comprehensive answer about AI agents"}
    ]
    plan_iter = iter(plan_sequence)

    async def mock_plan(*args, **kwargs):
        try:
            return next(plan_iter)
        except StopIteration:
            return {"type": "final", "output": "Fallback final"}

    agent.planner.plan = mock_plan

    result = await agent.run("Research AI agents")
    assert "Comprehensive answer" in result

    # Verify memory state
    assert len(agent.memory.episodic.get_all()) == 1
    semantic_hits = await agent.memory.semantic.search("AI agents", k=1)
    assert len(semantic_hits) == 1
