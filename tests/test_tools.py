# tests/test_tools.py
import asyncio
import json
from typing import Any, Dict
import pytest
from agentforge.core.tools import Tool, ToolRegistry, tool


# ---------------------------
# Test Fonksiyonları
# ---------------------------

def sync_add(a: int, b: int) -> int:
    """Adds two integers."""
    return a + b


async def async_multiply(x: float, y: float) -> float:
    """Multiplies two floats asynchronously."""
    await asyncio.sleep(0.01)
    return x * y


def failing_tool() -> str:
    raise RuntimeError("Simulated tool failure")


# ---------------------------
# Testler
# ---------------------------

class TestToolSchemaGeneration:
    def test_schema_from_sync_function(self):
        tool_obj = Tool("add", sync_add)
        schema = tool_obj.schema

        assert schema["name"] == "add"
        assert schema["description"] == "Adds two integers."
        assert schema["parameters"]["type"] == "object"
        assert "a" in schema["parameters"]["properties"]
        assert "b" in schema["parameters"]["properties"]
        assert schema["parameters"]["required"] == ["a", "b"]

        props = schema["parameters"]["properties"]
        assert props["a"]["type"] == "number"
        assert props["b"]["type"] == "number"

    def test_schema_from_async_function(self):
        tool_obj = Tool("multiply", async_multiply)
        schema = tool_obj.schema

        assert schema["name"] == "multiply"
        assert "float" in schema["description"]
        assert schema["parameters"]["required"] == ["x", "y"]
        props = schema["parameters"]["properties"]
        assert props["x"]["type"] == "number"
        assert props["y"]["type"] == "number"

    def test_schema_with_no_annotations(self):
        def no_annot(name, count):
            return f"{name}: {count}"

        tool_obj = Tool("no_annot", no_annot)
        schema = tool_obj.schema
        props = schema["parameters"]["properties"]
        # fallback to "string"
        assert props["name"]["type"] == "string"
        assert props["count"]["type"] == "string"


class TestToolExecution:
    @pytest.mark.asyncio
    async def test_sync_tool_execution_success(self):
        tool_obj = Tool("add", sync_add)
        result = await tool_obj.run({"a": 3, "b": 5})

        assert result["type"] == "tool"
        assert result["name"] == "add"
        assert result["success"] is True
        assert result["output"] == {"value": 8}

    @pytest.mark.asyncio
    async def test_async_tool_execution_success(self):
        tool_obj = Tool("multiply", async_multiply)
        result = await tool_obj.run({"x": 2.5, "y": 4.0})

        assert result["success"] is True
        assert result["output"] == {"value": 10.0}

    @pytest.mark.asyncio
    async def test_tool_execution_failure(self):
        tool_obj = Tool("failing", failing_tool)
        result = await tool_obj.run({})

        assert result["success"] is False
        assert "Simulated tool failure" in result["output"]["error"]

    @pytest.mark.asyncio
    async def test_tool_returns_dict(self):
        def dict_tool(query: str) -> Dict[str, Any]:
            return {"result": query.upper(), "length": len(query)}

        tool_obj = Tool("dict_tool", dict_tool)
        result = await tool_obj.run({"query": "hello"})

        assert result["success"] is True
        assert result["output"] == {"result": "HELLO", "length": 5}

    @pytest.mark.asyncio
    async def test_tool_returns_string(self):
        def str_tool(msg: str) -> str:
            return f"Echo: {msg}"

        tool_obj = Tool("str_tool", str_tool)
        result = await tool_obj.run({"msg": "test"})

        assert result["success"] is True
        assert result["output"] == {"text": "Echo: test"}


class TestToolDecorator:
    def test_tool_decorator_without_args(self):
        @tool
        def sample(x: int) -> int:
            """Doubles the input."""
            return x * 2

        assert isinstance(sample, Tool)
        assert sample.name == "sample"
        assert "Doubles" in sample.description

    def test_tool_decorator_with_custom_name_and_desc(self):
        @tool(name="doubler", description="Custom doubler")
        def my_func(val: int) -> int:
            return val * 2

        assert isinstance(my_func, Tool)
        assert my_func.name == "doubler"
        assert my_func.description == "Custom doubler"

    @pytest.mark.asyncio
    async def test_decorated_tool_execution(self):
        @tool
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        result = await greet.run({"name": "Alice"})
        assert result["success"] is True
        assert result["output"] == {"text": "Hello, Alice!"}


class TestToolRegistry:
    def test_register_tool(self):
        registry = ToolRegistry()
        t = Tool("add", sync_add)
        registry.add(t)
        assert registry.get("add") == t

    def test_register_duplicate_tool_raises_error(self):
        registry = ToolRegistry()
        t1 = Tool("add", sync_add)
        t2 = Tool("add", lambda x: x)
        registry.add(t1)
        with pytest.raises(ValueError, match="already registered"):
            registry.add(t2)

    def test_list_schemas_returns_dict(self):
        registry = ToolRegistry()
        registry.add(Tool("add", sync_add))
        schemas = registry.list_schemas()
        assert isinstance(schemas, dict)
        assert "add" in schemas
        assert schemas["add"]["name"] == "add"

    def test_get_nonexistent_tool_returns_none(self):
        registry = ToolRegistry()
        assert registry.get("nonexistent") is None


class TestJSONSchemaCompatibility:
    """
    Ensure generated schema matches OpenAI function calling spec.
    """

    def test_valid_openai_function_format(self):
        tool_obj = Tool("search", lambda q: f"Results for {q}")
        schema = tool_obj.schema

        # Must have these top-level keys
        assert "name" in schema
        assert "description" in schema
        assert "parameters" in schema

        params = schema["parameters"]
        assert params["type"] == "object"
        assert isinstance(params["properties"], dict)
        assert isinstance(params.get("required", []), list)

        # Try serializing to JSON (should not fail)
        json.dumps(schema)


# ---------------------------
# Opsiyonel: Entegrasyon Testi (Agent ile uyum)
# ---------------------------

@pytest.mark.asyncio
async def test_tool_integration_with_mock_agent():
    """
    Simüle edilmiş bir agent döngüsünde tool kullanımı.
    """
    tools = [Tool("add", sync_add), Tool("multiply", async_multiply)]
    registry = ToolRegistry()
    for t in tools:
        registry.add(t)

    # Simulate agent calling planner output
    plan_output = {"type": "tool", "name": "add", "input": {"a": 10, "b": 20}}

    tool_name = plan_output["name"]
    tool_input = plan_output["input"]
    selected_tool = registry.get(tool_name)

    result = await selected_tool.run(tool_input)
    assert result["success"] is True
    assert result["output"]["value"] == 30



    
