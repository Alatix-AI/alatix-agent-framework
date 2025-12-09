# agentforge/core/orchestrator.py
import asyncio
import json
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

class SubAgentOrchestrator:
    def __init__(self, agents: List, workspace_dir: Optional[str] = None):
        self.agents = {agent.name: agent for agent in agents}
        self.agent_list = list(agents)
        self.workspace = Path(workspace_dir) if workspace_dir else Path(".agentforge/workspace")
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.max_replan_attempts = 3

    async def run(self, task: str) -> str:
        plan = await self._decompose_task(task)
        attempt = 0

        while attempt < self.max_replan_attempts:
            attempt += 1
            try:
                result = await self._execute_plan(task, plan)
                if result.get("success"):
                    return result["output"]
                else:
                    # yeniden planla
                    feedback = result.get("feedback", "")
                    plan = await self._replan(task, plan, feedback)
            except Exception as e:
                # fallback replan
                plan = await self._replan(task, plan, f"Critical error: {str(e)}")

        # Son çare: ilk ajanla tek başına çöz
        first_agent = self.agent_list[0]
        return await first_agent.run(task)

    async def _execute_plan(self, task: str, plan: List[Dict[str, str]]) -> Dict[str, Any]:
        # Her adım sonrası inbox'lar temizlenir
        for agent in self.agent_list:
            agent.inbox.clear()

        for step in plan:
            agent_name = step["agent"]
            sub_task = step["task"]

            if agent_name not in self.agents:
                continue

            agent = self.agents[agent_name]

            # Diğer ajanlardan gelen mesajları görevle birleştir
            inbox_summary = agent.get_inbox_summary()
            if inbox_summary != "(no messages)":
                enhanced_task = f"{sub_task}\n\nCRITICAL CONTEXT FROM PEERS:\n{inbox_summary}"
            else:
                enhanced_task = sub_task

            try:
                result_text = ""
                
                # <<< YENİ KOD BAŞI: STREAM'DEN EVENT OKU >>>
                async for event in agent.run_stream(enhanced_task):
                    if event["type"] == "tool":
                        # Tool event'ini özetle ve diğer ajanlara yayınla
                        tool_summary = await self._summarize_tool_event(agent_name, event)
                        for other_agent in self.agent_list:
                            if other_agent.name != agent_name:
                                other_agent.receive_message(
                                    sender=agent_name,
                                    content=tool_summary,
                                    priority="info"
                                )
                    elif event["type"] == "final":
                       result_text = event["text"]
                # <<< YENİ KOD SONU >>>
                # Sonucu diğer ajanlara kritik mesaj olarak yayınla
                summary = await self._summarize_for_peers(agent_name, enhanced_task, result_text)
                for other_agent in self.agent_list:
                    if other_agent.name != agent_name:
                        other_agent.receive_message(
                            sender=agent_name,
                            content=summary,
                            priority="info"
                        )

                # Hata tespiti: sonuçta "error", "failed", "impossible" geçiyorsa
                if any(word in result_text.lower() for word in ["error", "failed", "impossible", "deprecated", "not found"]):
                    return {
                        "success": False,
                        "feedback": f"Agent '{agent_name}' reported issue: {result_text[:200]}"
                    }

            except Exception as e:
                # Hata durumunda kritik mesaj yayınla
                for other_agent in self.agent_list:
                    if other_agent.name != agent_name:
                        other_agent.receive_message(
                            sender=agent_name,
                            content=f"FAILED: {str(e)}",
                            priority="error"
                        )
                return {
                    "success": False,
                    "feedback": f"Agent '{agent_name}' crashed: {str(e)}"
                }

        # Son sentez
        synthesis_prompt = f"""
        FINAL SYNTHESIS TASK: {task}

        Agent Outputs:
        """
        for agent in self.agent_list:
            # En son görev sonucunu tahmin et (bellekten son episodik eylemi al)
            episodic = agent.memory.episodic.search_last(k=1)
            if episodic:
                last_output = episodic[0].get("result", "")
            else:
                last_output = "[no output]"
            synthesis_prompt += f"\n{agent.name}:\n{last_output}\n"

        first_llm = self.agent_list[0].llm
        final_resp = await first_llm.generate(synthesis_prompt)
        return {"success": True, "output": final_resp.text}

    async def _decompose_task(self, task: str) -> List[Dict[str, str]]:
        agent_names = ", ".join(self.agents.keys())
        prompt = f"""
        You are a task decomposer. Break this task into sequential subtasks.
        Assign each to one agent: {agent_names}.

        TASK: {task}

        Output JSON list: [{{"agent": "Name", "task": "description"}}]
        """
        first_llm = self.agent_list[0].llm
        raw = await first_llm.generate(prompt)
        try:
            return json.loads(raw.text)
        except:
            return [{"agent": list(self.agents.keys())[0], "task": task}]

    async def _replan(self, task: str, old_plan: List[Dict[str, str]], feedback: str) -> List[Dict[str, str]]:
        agent_names = ", ".join(self.agents.keys())
        prompt = f"""
        You are replanning due to failure.

        ORIGINAL TASK: {task}
        PREVIOUS PLAN: {json.dumps(old_plan, indent=2)}
        FAILURE FEEDBACK: {feedback}

        Create a NEW plan that addresses the issue.
        Output JSON list: [{{"agent": "Name", "task": "description"}}]
        """
        first_llm = self.agent_list[0].llm
        raw = await first_llm.generate(prompt)
        try:
            return json.loads(raw.text)
        except:
            # fallback: ilk ajan tüm görevi alsın
            return [{"agent": list(self.agents.keys())[0], "task": task}]

    async def _summarize_for_peers(self, sender: str, task: str, output: str) -> str:
        """
        Agent'ın çıktısını diğer ajanlar için 1-2 cümleyle özetle.
        """
        prompt = f"""
        Summarize this agent's work in 1-2 sentences for other agents.

        AGENT: {sender}
        TASK: {task[:300]}
        OUTPUT: {output[:500]}

        Summary:
        """
        first_llm = self.agent_list[0].llm
        raw = await first_llm.generate(prompt, temperature=0.3)
        return raw.text.strip()
    
    
    async def _summarize_tool_event(self, agent_name: str, event: Dict[str, Any]) -> str:
        """
        Bir tool event'ini diğer ajanlar için 1 cümleye indirge.
        """
        tool_name = event.get("name", "unknown")
        tool_input = event.get("input", {})
        tool_result = event.get("result", {})

        # Basit metin özet (LLM'e ihtiyaç duyulmayabilir, ama tutarlılık için kullanalım)
        input_str = json.dumps(tool_input, ensure_ascii=False)[:200]
        if isinstance(tool_result, dict) and tool_result.get("success"):
            output_str = str(tool_result.get("output", ""))[:300]
            prompt = f"Tool '{tool_name}' by {agent_name}:\nInput: {input_str}\nOutput: {output_str}\n\nSummarize in one sentence for other agents:"
        else:
            error_msg = str(tool_result.get("output", {}).get("error", "unknown error"))[:200]
            prompt = f"Tool '{tool_name}' by {agent_name} FAILED:\nInput: {input_str}\nError: {error_msg}\n\nBriefly state what failed:"

        try:
            raw = await self.agent_list[0].llm.generate(prompt, temperature=0.2)
            return raw.text.strip()
        except:
            # Fallback: direkt metin
            if tool_result.get("success"):
                return f"[{agent_name}] Used '{tool_name}' → got result"
            else:
                return f"[{agent_name}] '{tool_name}' failed"
