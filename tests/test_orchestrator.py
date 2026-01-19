# tests/test_orchestrator.py
import asyncio
import json
import pytest
from unittest.mock import AsyncMock

from core.orchestrator import SubAgentOrchestrator
from core.agent import Agent
from core.tools import Tool


# ---------------------------
# Mock Tools
# ---------------------------

def mock_search(query: str) -> str:
    return f"Results for {query}"


def mock_code(task: str) -> str:
    return f"Generated code for: {task}"


# ---------------------------
# Mock Agents
# ---------------------------

class MockAgent:
    def __init__(self, name: str, tool_func=None):
        self.name = name
        self.inbox = []
        self.memory = AsyncMock()
        self.memory.episodic.search_last.return_value = [{"result": "mock result"}]
        self.llm = AsyncMock()
        self.llm.generate.return_value = "Synthesized output"
        if tool_func:
            self.tool = Tool(name + "_tool", tool_func)
        else:
            self.tool = None

    async def run_stream(self, task: str):
        # Simulate tool event then final
        if "search" in task.lower():
            yield {
                "type": "tool",
                "name": "search_tool",
                "input": {"query": "AI agents"},
                "result": {"success": True, "output": {"text": "Found key papers"}}
            }
        elif "code" in task.lower():
            yield {
                "type": "tool",
                "name": "code_tool",
                "input": {"task": "agent framework"},
                "result": {"success": True, "output": {"text": "class Agent: ..."}}
            }
        yield {"type": "final", "text": f"Completed {task}"}

    def receive_message(self, sender: str, content: str, priority: str = "info"):
        self.inbox.append({"from": sender, "content": content, "priority": priority})

    def get_inbox_summary(self, max_items: int = 5) -> str:
        if not self.inbox:
            return "(no messages)"
        return "\n".join([f"[{msg['from']}]: {msg['content']}" for msg in self.inbox[:max_items]])


# ---------------------------
# Test Fixtures
# ---------------------------

@pytest.fixture
def mock_agents():
    researcher = MockAgent("Researcher", mock_search)
    coder = MockAgent("Coder", mock_code)
    return [researcher, coder]


@pytest.fixture
def orchestrator(mock_agents):
    return SubAgentOrchestrator(mock_agents, workspace_dir=".test_workspace")


# ---------------------------
# Core Tests
# ---------------------------

class TestTaskDecomposition:
    @pytest.mark.asyncio
    async def test_decompose_task(self, orchestrator):
        # Mock LLM response
        orchestrator.agent_list[0].llm.generate.return_value = json.dumps([
            {"agent": "Researcher", "task": "Find latest AI agent papers"},
            {"agent": "Coder", "task": "Implement agent framework"}
        ])

        plan = await orchestrator._decompose_task("Build an AI agent system")
        assert len(plan) == 2
        assert plan[0]["agent"] == "Researcher"
        assert "papers" in plan[0]["task"]


class TestExecutionFlow:
    @pytest.mark.asyncio
    async def test_execute_plan_with_inbox_communication(self, orchestrator):
        # Mock decomposition
        orchestrator._decompose_task = AsyncMock(return_value=[
            {"agent": "Researcher", "task": "Search for AI agents"},
            {"agent": "Coder", "task": "Write code based on findings"}
        ])

        # Mock LLM for summarization
        orchestrator.agent_list[0].llm.generate = AsyncMock(return_value="Research found key frameworks")

        result = await orchestrator._execute_plan("Build agent", [
            {"agent": "Researcher", "task": "Search for AI agents"},
            {"agent": "Coder", "task": "Write code based on findings"}
        ])

        # Both agents should have exchanged messages
        researcher, coder = orchestrator.agent_list
        assert len(coder.inbox) >= 1  # Coder received from Researcher
        assert "Research found" in coder.inbox[0]["content"]

        assert result["success"] is True
        assert "Synthesized output" in result["output"]


class TestToolEventSharing:
    @pytest.mark.asyncio
    async def test_tool_events_shared_as_peer_messages(self, mock_agents):
        orchestrator = SubAgentOrchestrator(mock_agents)

        # Mock plan
        plan = [{"agent": "Researcher", "task": "Search for AI agents"}]

        result = await orchestrator._execute_plan("Research task", plan)

        # Coder should have received tool event from Researcher
        coder = mock_agents[1]
        assert len(coder.inbox) > 0
        message = coder.inbox[0]["content"]
        assert "Tool 'search_tool'" in message or "Used 'search_tool'" in message


class TestReplanningOnFailure:
    @pytest.mark.asyncio
    async def test_replan_on_agent_failure(self, mock_agents):
        orchestrator = SubAgentOrchestrator(mock_agents)

        # Mock initial plan that fails
        orchestrator._decompose_task = AsyncMock(return_value=[
            {"agent": "Researcher", "task": "Impossible task"}
        ])

        # Mock Researcher to return error
        original_run_stream = mock_agents[0].run_stream
        async def failing_run_stream(task):
            yield {"type": "final", "text": "Error: impossible task failed"}

        mock_agents[0].run_stream = failing_run_stream

        # Mock replan response
        mock_agents[0].llm.generate = AsyncMock(return_value=json.dumps([
            {"agent": "Researcher", "task": "Try alternative approach"}
        ]))

        result = await orchestrator.run("Impossible task")
        assert result is not None
        # Should have attempted replan
        assert mock_agents[0].llm.generate.call_count >= 2


class TestSynthesis:
    @pytest.mark.asyncio
    async def test_final_synthesis_includes_all_agents(self, orchestrator):
        # Mock both agents' episodic memory
        for agent in orchestrator.agent_list:
            agent.memory.episodic.search_last.return_value = [{"result": f"{agent.name} result"}]

        synthesis_prompt = """
        FINAL SYNTHESIS TASK: Build agent

        Agent Outputs:

        Researcher:
        Researcher result

        Coder:
        Coder result
        """

        # Mock LLM synthesis
        orchestrator.agent_list[0].llm.generate = AsyncMock(return_value="Final synthesized answer")

        result = await orchestrator._execute_plan("Build agent", [
            {"agent": "Researcher", "task": "Research"},
            {"agent": "Coder", "task": "Code"}
        ])

        assert result["success"] is True
        assert "Final synthesized answer" in result["output"]
        # Verify synthesis prompt included both agents
        call_args = orchestrator.agent_list[0].llm.generate.call_args
        assert "Researcher result" in call_args[0][0]
        assert "Coder result" in call_args[0][0]


class TestMultiPassExecution:
    @pytest.mark.asyncio
    async def test_max_replan_attempts_respected(self, mock_agents):
        orchestrator = SubAgentOrchestrator(mock_agents, workspace_dir=".test")
        orchestrator.max_replan_attempts = 2

        # Always fail
        async def always_fail_run_stream(task):
            yield {"type": "final", "text": "Failed"}

        for agent in mock_agents:
            agent.run_stream = always_fail_run_stream

        # Mock decompose and replan to also fail
        orchestrator._decompose_task = AsyncMock(return_value=[{"agent": "Researcher", "task": "Fail"}])
        orchestrator._replan = AsyncMock(return_value=[{"agent": "Researcher", "task": "Fail again"}])

        result = await orchestrator.run("Failing task")
        # Should fall back to first agent running alone
        assert result is not None
        assert orchestrator._replan.call_count <= 2


# ---------------------------
# Integration Test: Full Flow
# ---------------------------

@pytest.mark.asyncio
async def test_orchestrator_full_integration_flow():
    # Create realistic mock agents
    researcher = MockAgent("Researcher", mock_search)
    coder = MockAgent("Coder", mock_code)

    orchestrator = SubAgentOrchestrator([researcher, coder])

    # Mock decomposition
    researcher.llm.generate = AsyncMock(return_value=json.dumps([
        {"agent": "Researcher", "task": "Search for AI agent frameworks"},
        {"agent": "Coder", "task": "Implement a simple agent using findings"}
    ]))

    # Mock synthesis
    researcher.llm.generate.side_effect = [
        json.dumps([{"agent": "Researcher", "task": "Search..."}, {"agent": "Coder", "task": "Implement..."}]),
        "Final integrated agent system built successfully"
    ]

    result = await orchestrator.run("Build an AI agent system")

    assert result is not None
    assert "successfully" in result
    # Verify communication happened
    assert len(coder.inbox) > 0
    assert len(researcher.inbox) >= 0  # May receive from Coder too
