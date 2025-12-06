# agentforge/core/orchestrator.py
from typing import List, Dict, Any, Optional
import asyncio
import json
import os
from pathlib import Path

from agentforge.core.agent import Agent

class SubAgentOrchestrator:
    def __init__(
        self,
        agents: List[Agent],  # ← Döngüsel import için string kullan
        workspace_dir: Optional[str] = None
    ):
        self.agents = {agent.name: agent for agent in agents}
        self.workspace = Path(workspace_dir) if workspace_dir else Path(".agentforge/workspace")
        self.workspace.mkdir(parents=True, exist_ok=True)

    async def run(self, task: str) -> str:
        """
        Ana görevi analizle, sub-agent'leri çalıştır, sonuçları sentezle.
        """
        # 1. Görevi analizle: hangi agent'ler ne yapmalı?
        plan = await self._decompose_task(task)
        
        # 2. Sub-agent'leri sırayla çalıştır
        for step in plan:
            agent_name = step["agent"]
            sub_task = step["task"]
            agent = self.agents[agent_name]
            
            # Agent'a özel bir workspace dosyası yolla
            output_file = self.workspace / f"{agent_name}_output.txt"
            if output_file.exists():
                output_file.unlink()  # temizle

            # Sub-agent görevi çalıştır
            result = await agent.run(sub_task)
            
            # Sonucu dosyaya kaydet (diğer agent'ler okuyabilir)
            output_file.write_text(result)

        # 3. Sonuçları oku ve sentezle
        results = {}
        for name in self.agents:
            file = self.workspace / f"{name}_output.txt"
            if file.exists():
                results[name] = file.read_text()
            else:
                results[name] = ""

        # Ana sentez görevi
        synthesis_prompt = f"""
        You are a synthesis agent. Combine the following sub-agent results into a coherent final answer.

        TASK: {task}

        SUB-AGENT RESULTS:
        {json.dumps(results, indent=2)}

        Provide a clear, concise final response.
        """
        # Ana agent olarak kendi LLM'ini kullan (veya ilk sub-agent'in LLM'ini)
        first_agent = list(self.agents.values())[0]
        final_result = await first_agent.llm.generate(synthesis_prompt)
        return final_result.text

    async def _decompose_task(self, task: str) -> List[Dict[str, str]]:
        """
        Görevi sub-agent adımlarına ayır.
        Örnek çıktı:
        [
            {"agent": "Researcher Agent", "task": "Search latest AI Agent advancements"},
            {"agent": "Coding Agent", "task": "Build data agent using crewai"}
        ]
        """
        agent_names = ", ".join(self.agents.keys())
        prompt = f"""
        You are a task orchestrator. Decompose the following task into sequential subtasks.
        Assign each subtask to one of these agents: {agent_names}.

        TASK: {task}

        Respond with a JSON list of objects:
        [{{"agent": "<agent-name>", "task": "<subtask>"}}]
        """
        first_agent = list(self.agents.values())[0]
        raw = await first_agent.llm.generate(prompt)
        try:
            plan = json.loads(raw.text)
            return plan
        except:
            # Fallback: İlk agent her şeyi yapar
            return [{"agent": list(self.agents.keys())[0], "task": task}]