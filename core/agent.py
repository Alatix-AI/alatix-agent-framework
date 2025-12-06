# atlatix/core/agent.py
from __future__ import annotations
import json
from typing import List, Dict, Any, Optional
import asyncio

from agentforge.core.llm import LLMAdapter
from agentforge.core.tools import Tool, ToolRegistry
from agentforge.core.memory import MemorySystem
from agentforge.core.planner import AdvancedPlanner


class Agent:
    def __init__(
        self,
        model: str = "openai:gpt-4o-mini",
        tools: Optional[List[Tool]] = None,
        memory: Optional[MemorySystem] = None,
        max_steps: int = 3,
        semantic_k: int = 2,
        episodic_k: int = 2,
        max_tokens: Optional[int] = None,  
        temperature: Optional[float] = None,  
        api_key: Optional[str] = None,
        persistent: bool = False,
        name: str = None,
        **llm_kwargs
    ):
        self.model = model
        self.name = name or f"Agent-{id(self):x}"

        if max_tokens is not None:
            llm_kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            llm_kwargs["temperature"] = temperature   
        if api_key is not None:
            llm_kwargs["api_key"] = api_key
          
        self.llm = LLMAdapter(model, **llm_kwargs)

        self.tool_registry = ToolRegistry()
        for t in (tools or []):
            self.tool_registry.add(t)

        self.memory = memory or MemorySystem(self.llm, persistent=persistent, name=self.name)

        planner_temp = temperature if temperature is not None else None
        self.planner = AdvancedPlanner(
            llm_adapter=self.llm, 
            max_retries=2,
            candidate_temperature=planner_temp, 
            eval_temperature=planner_temp
        )
        self.max_steps = max_steps

        # retrieval params
        self.semantic_k = semantic_k
        self.episodic_k = episodic_k

    def register_tool(self, tool: Tool):
        self.tool_registry.add(tool)

    # -------------------------
    # Non-streaming wrapper using run_stream
    # -------------------------
    async def run(self, task: str) -> str:
        final_text = ""
        async for ev in self.run_stream(task):
            if ev["type"] == "token":
                final_text += ev["text"]
            elif ev["type"] == "final":
                if ev.get("text"):
                    final_text = ev["text"]
        return final_text

    # -------------------------
    # Streaming main loop
    # -------------------------
    async def run_stream(self, task: str):
        """
        Yields events:
        - {"type":"token","text": "..."} for streamed final outputs
        - {"type":"tool", "name":..., "input":..., "result":...} after tool execution
        - {"type":"final","text": "..."} when finished
        """
        history: List[Dict[str, Any]] = []

        for step in range(self.max_steps):
            # --- retrieve context from memory (RAG) ---
            context = await self.memory.retrieve_context(
                task, k_semantic=self.semantic_k, k_episodic=self.episodic_k
            )

            # --- ask planner for next step ---
            plan = await self.planner.plan(
                task,
                history,
                context=context,
                tools=self.tool_registry.list_schemas()
            )

            # --- FINAL RESPONSE ---
            if plan.get("type") == "final":
                final_text = plan.get("output", "")
                streamed_text = ""

                # Doğrudan final_text'i akışa al (LLM'e tekrar sorma!)
                for token in final_text.split():  # basit kelime bazlı streaming
                    token_with_space = token + " "
                    streamed_text += token_with_space
                    yield {"type": "token", "text": token_with_space}

                # persist final to semantic memory
                await self.memory.add_knowledge(
                    text=streamed_text.strip(),
                    metadata={"source": "final_output", "task": task}
                )

                yield {"type": "final", "text": streamed_text.strip()}
                return

            # --- TOOL EXECUTION ---
            elif plan.get("type") == "tool":
                tool_name = plan.get("name")
                args = plan.get("input") or {}

                tool = self.tool_registry.get(tool_name)
                if not tool:
                    raise RuntimeError(f"AgentError: Tool '{tool_name}' not found")

                # async/sync safe execution
                if asyncio.iscoroutinefunction(tool.func):
                    result = await tool.run(args)
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, lambda: tool.run(args))

                # --- memory kayıtları ---
                history.append({"action": {"tool": tool_name, "input": args}, "result": result})
                await self.memory.add_episode(action=tool_name, result=result)

                # semantic memory için normalize
                output_text = ""
                if result.get("success"):
                    if isinstance(result["output"], dict):
                        output_text = json.dumps(result["output"], ensure_ascii=False)
                    else:
                        output_text = str(result["output"])
                    await self.memory.add_knowledge(
                        text=output_text,
                        metadata={"source": f"tool:{tool_name}", "input": args}
                    )

                # agent → UI event olarak dön
                yield {"type": "tool", "name": tool_name, "input": args, "result": result}

            else:
                # dead-end veya bilinmeyen tip → history döndür
                yield {"type": "final", "text": json.dumps(history, indent=2, ensure_ascii=False)}
                return

        # max_steps aşıldıysa history özet
        yield {"type": "final", "text": json.dumps(history, indent=2, ensure_ascii=False)}
 
