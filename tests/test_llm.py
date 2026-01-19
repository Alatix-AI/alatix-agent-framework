# tests/test_llm.py
import asyncio
import json
import os
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from core.llm import (
    LLMAdapter,
    OpenAIAdapter,
    AnthropicAdapter,
    GeminiAdapter,
    OllamaAdapter,
    ReplicateAdapter,
    HuggingFaceAdapter,
    LLMStats,
    count_tokens
)


# ---------------------------
# Mock Responses
# ---------------------------

class MockOpenAIResponse:
    def __init__(self, text="Mock response", tokens=10):
        self.choices = [MagicMock(message=MagicMock(content=text))]
        self.usage = MagicMock(total_tokens=tokens)


class MockAnthropicResponse:
    def __init__(self, text="Mock response"):
        self.content = [MagicMock(text=text)]
        self.usage = MagicMock(input_tokens=5, output_tokens=5)


class MockGeminiResponse:
    def __init__(self, text="Mock response"):
        self.text = text


# ---------------------------
# Test Fixtures
# ---------------------------

@pytest.fixture
def mock_openai_client():
    with patch('openai.AsyncOpenAI') as mock:
        mock_instance = AsyncMock()
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_anthropic_client():
    with patch('anthropic.AsyncAnthropic') as mock:
        mock_instance = AsyncMock()
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_gemini_client():
    with patch('google.genai.Client') as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock_instance


# ---------------------------
# LLMAdapter Core Tests
# ---------------------------

class TestLLMAdapterCore:
    @pytest.mark.asyncio
    async def test_generate_with_cache(self):
        # Mock adapter
        mock_adapter = AsyncMock()
        mock_adapter.generate.return_value = "Cached response"

        with patch('agentforge.core.llm.LLMClient._make_adapter', return_value=mock_adapter):
            adapter = LLMAdapter("mock:test")
            result1 = await adapter.generate("test prompt")
            result2 = await adapter.generate("test prompt")  # Should use cache

            assert result1 == result2 == "Cached response"
            assert mock_adapter.generate.call_count == 1  # Second call cached

    @pytest.mark.asyncio
    async def test_stream_basic(self):
        mock_adapter = AsyncMock()
        mock_adapter.stream = AsyncMock(return_value=["token1 ", "token2 "])

        with patch('agentforge.core.llm.LLMClient._make_adapter', return_value=mock_adapter):
            adapter = LLMAdapter("mock:test")
            tokens = []
            async for token in adapter.stream("test"):
                tokens.append(token)
            
            assert tokens == ["token1 ", "token2 "]

    @pytest.mark.asyncio
    async def test_embed_basic(self):
        mock_adapter = AsyncMock()
        mock_adapter.embed.return_value = [[0.1, 0.2, 0.3]]

        with patch('agentforge.core.llm.LLMClient._make_adapter', return_value=mock_adapter):
            adapter = LLMAdapter("mock:test")
            embeddings = await adapter.embed(["test text"])
            assert embeddings == [[0.1, 0.2, 0.3]]


# ---------------------------
# OpenAI Adapter Tests
# ---------------------------

class TestOpenAIAdapter:
    @pytest.mark.asyncio
    async def test_generate_success(self, mock_openai_client):
        mock_openai_client.chat.completions.create.return_value = MockOpenAIResponse("Hello world", 15)
        
        adapter = OpenAIAdapter("openai:gpt-4o-mini", api_key="test-key")
        result = await adapter.generate("Say hello")
        
        assert result.text == "Hello world"
        assert result.tokens_used == 15
        mock_openai_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_with_functions(self, mock_openai_client):
        mock_response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "function_call": {
                        "name": "search",
                        "arguments": '{"query": "AI"}'
                    }
                },
                "finish_reason": "function_call"
            }]
        }
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        adapter = OpenAIAdapter("openai:gpt-4o-mini", api_key="test-key")
        functions = [{"name": "search", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}]
        result = await adapter.generate_with_functions("Search for AI", functions)
        
        assert "choices" in result
        assert result["choices"][0]["message"]["function_call"]["name"] == "search"


# ---------------------------
# Anthropic Adapter Tests
# ---------------------------

class TestAnthropicAdapter:
    @pytest.mark.asyncio
    async def test_generate_success(self, mock_anthropic_client):
        mock_anthropic_client.messages.create.return_value = MockAnthropicResponse("Claude says hello")
        
        adapter = AnthropicAdapter("anthropic:claude-3-haiku", api_key="test-key")
        result = await adapter.generate("Say hello")
        
        assert result.text == "Claude says hello"
        assert result.tokens_used == 10

    @pytest.mark.asyncio
    async def test_generate_with_functions_openai_format(self, mock_anthropic_client):
        # Mock Anthropic response with tool use
        mock_tool_use = MagicMock(type="tool_use", name="search", input={"query": "AI"})
        mock_text = MagicMock(type="text", text="")
        mock_response = MagicMock()
        mock_response.content = [mock_tool_use]
        
        mock_anthropic_client.messages.create.return_value = mock_response
        
        adapter = AnthropicAdapter("anthropic:claude-3-haiku", api_key="test-key")
        functions = [{"name": "search", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}]
        result = await adapter.generate_with_functions("Search for AI", functions)
        
        # Should be converted to OpenAI format
        assert "choices" in result
        choice = result["choices"][0]
        assert choice["message"]["function_call"]["name"] == "search"
        assert "AI" in choice["message"]["function_call"]["arguments"]


# ---------------------------
# Gemini Adapter Tests
# ---------------------------

class TestGeminiAdapter:
    @pytest.mark.asyncio
    async def test_generate_success(self, mock_gemini_client):
        mock_response = MockGeminiResponse("Gemini response")
        mock_gemini_client.models.generate_content.return_value = mock_response
        
        with patch('google.genai.types') as mock_types:
            mock_types.GenerateContentConfig.return_value = {}
            adapter = GeminiAdapter("gemini:gemini-1.5-flash", api_key="test-key")
            result = await adapter.generate("Say hello")
            
            assert result.text == "Gemini response"

    @pytest.mark.asyncio
    async def test_generate_with_functions_openai_format(self, mock_gemini_client):
        # Mock Gemini response with function call
        mock_function_call = MagicMock()
        mock_function_call.name = "search"
        mock_function_call.args = {"query": "AI"}
        mock_part = MagicMock()
        mock_part.function_call = mock_function_call
        mock_content = MagicMock()
        mock_content.parts = [mock_part]
        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        
        mock_gemini_client.models.generate_content.return_value = mock_response
        
        with patch('google.genai.types') as mock_types:
            mock_types.GenerateContentConfig.return_value = {}
            mock_types.FunctionDeclaration.return_value = {}
            mock_types.Tool.return_value = {}
            mock_types.Schema.return_value = {}
            
            adapter = GeminiAdapter("gemini:gemini-1.5-flash", api_key="test-key")
            functions = [{"name": "search", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}]
            result = await adapter.generate_with_functions("Search for AI", functions)
            
            # Should be converted to OpenAI format
            assert "choices" in result
            choice = result["choices"][0]
            assert choice["message"]["function_call"]["name"] == "search"
            assert "AI" in choice["message"]["function_call"]["arguments"]


# ---------------------------
# Ollama & Replicate Tests (Basic)
# ---------------------------

class TestOllamaAdapter:
    @pytest.mark.asyncio
    async def test_generate_success(self):
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = AsyncMock()
            mock_response.json.return_value = {"output": "Ollama response"}
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response
            
            adapter = OllamaAdapter("ollama:llama3", base_url="http://localhost:11434")
            result = await adapter.generate("Say hello")
            
            assert result.text == "Ollama response"


class TestReplicateAdapter:
    @pytest.mark.asyncio
    async def test_generate_success(self):
        with patch('replicate.Client') as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_instance.run.return_value = ["Replicate ", "response"]
            
            adapter = ReplicateAdapter("replicate:meta/llama3", api_key="test-key")
            result = await adapter.generate("Say hello")
            
            assert result.text == "Replicate response"


# ---------------------------
# HuggingFace Adapter Tests
# ---------------------------

class TestHuggingFaceAdapter:
    @pytest.mark.asyncio
    async def test_generate_success(self):
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = AsyncMock()
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "HF response"}}]
            }
            mock_response.status_code = 200
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response
            
            adapter = HuggingFaceAdapter(
                "huggingface:meta/llama3", 
                api_key="test-key"
            )
            result = await adapter.generate("Say hello")
            
            assert result.text == "HF response"


# ---------------------------
# Stats and Utility Tests
# ---------------------------

class TestLLMStats:
    def test_stats_accumulation(self):
        stats = LLMStats()
        stats.calls = 1
        stats.tokens = 100
        stats.cost_usd = 0.01
        
        # Simulate another call
        stats.calls += 1
        stats.tokens += 50
        stats.cost_usd += 0.005
        
        assert stats.calls == 2
        assert stats.tokens == 150
        assert stats.cost_usd == 0.015


class TestTokenCounting:
    def test_count_tokens(self):
        if "tiktoken" not in globals():
            pytest.skip("tiktoken not available")
        
        count = count_tokens("gpt-3.5-turbo", "Hello world!")
        assert isinstance(count, int)
        assert count > 0


# ---------------------------
# Error Handling Tests
# ---------------------------

class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_adapter_error_propagation(self):
        with patch('openai.AsyncOpenAI') as mock:
            mock_instance = AsyncMock()
            mock_instance.chat.completions.create.side_effect = Exception("API Error")
            mock.return_value = mock_instance
            
            adapter = OpenAIAdapter("openai:gpt-4o-mini", api_key="test-key")
            with pytest.raises(RuntimeError, match="API Error"):
                await adapter.generate("test")


# ---------------------------
# Multi-LLM Manager Tests
# ---------------------------

@pytest.mark.asyncio
async def test_multi_llm_manager_first_strategy():
    """Test MultiLLMManager with 'first' strategy"""
    from agentforge.core.llm import MultiLLMManager
    
    mock_adapter1 = AsyncMock()
    mock_adapter1.generate.side_effect = Exception("First failed")
    
    mock_adapter2 = AsyncMock()
    mock_adapter2.generate.return_value = "Success from second"
    
    manager = MultiLLMManager([mock_adapter1, mock_adapter2])
    result = await manager.generate("test", strategy="first")
    
    assert result == "Success from second"
    assert mock_adapter1.generate.called
    assert mock_adapter2.generate.called


@pytest.mark.asyncio
async def test_multi_llm_manager_ensemble_strategy():
    """Test MultiLLMManager with 'ensemble' strategy"""
    from agentforge.core.llm import MultiLLMManager
    
    mock_adapter1 = AsyncMock()
    mock_adapter1.generate.return_value = "Response 1"
    
    mock_adapter2 = AsyncMock()
    mock_adapter2.generate.return_value = "Response 2"
    
    manager = MultiLLMManager([mock_adapter1, mock_adapter2])
    result = await manager.generate("test", strategy="ensemble")
    
    assert "Response 1" in result
    assert "Response 2" in result
