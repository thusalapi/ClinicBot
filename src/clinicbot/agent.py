"""The agent loop: the core of every "AI agent".

    while not done:
        reply = LLM(conversation, tools)
        if reply has tool calls:
            run each tool (asking a human first if it changes anything)
            append the results to the conversation
        else:
            return reply.text          # the model decided it's finished

Everything else (frameworks, platforms, "agentic" products) is built around
this loop. Things worth noticing in the code:
  - MAX_STEPS stops a confused model from looping forever (and burning money
    on a paid API).
  - The conversation (self.messages) is the agent's only memory. Each turn
    resends the whole history, including old tool results.
  - Approval for side effects lives in OUR code, not in the prompt. A prompt
    can be ignored or jailbroken; an `if` statement can't.
"""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta

import ollama

from clinicbot.config import AGENT_THINK, OLLAMA_MODEL
from clinicbot.tools import SIDE_EFFECT_TOOLS, TOOL_SCHEMAS, run_tool

MAX_STEPS = 8


def _calendar() -> str:
    # Small models are bad at date arithmetic ("next Tuesday" = ?), so we do it
    # for them and put a ready-made calendar in the prompt.
    days = [date.today() + timedelta(days=i) for i in range(14)]
    return "\n".join(f"- {d:%A} {d.isoformat()}" for d in days)


def system_prompt() -> str:
    return f"""\
You are ClinicBot, the front-desk assistant for Sunrise Family Clinic.
Today is {date.today():%A} {date.today().isoformat()}. The next 14 days:
{_calendar()}

How to work:
- For questions about the clinic (hours, fees, policies, tests, vaccines), call \
search_clinic_info and answer only from what it returns.
- To book: find the right doctor (list_doctors if unsure), call check_availability, \
offer a few free slots, and get the patient's full name and phone number. Then call \
book_appointment. The system will ask the patient to approve; don't ask them yourself.
- Never invent slots, doctor ids or appointment ids. Use only values from tool results.
- If a tool returns an error, read it, fix the problem or ask the patient, and try again.
- For emergencies (chest pain, difficulty breathing, severe bleeding, stroke signs) tell \
the patient to call 1990 immediately. Do not give medical diagnoses.
- Keep replies short and friendly."""


@dataclass
class ToolCallRecord:
    name: str
    arguments: dict
    result: dict


@dataclass
class Agent:
    # Called before any side-effect tool runs; return True to allow it.
    approve: Callable[[str, dict], bool]
    # Called with a line of text for every tool call/result (None = silent).
    trace: Callable[[str], None] | None = None
    messages: list = field(default_factory=list)
    tool_log: list[ToolCallRecord] = field(default_factory=list)

    def __post_init__(self):
        self.messages = [{"role": "system", "content": system_prompt()}]

    def send(self, user_text: str) -> str:
        self.messages.append({"role": "user", "content": user_text})
        turn_start = len(self.tool_log)
        nudged = False

        for _ in range(MAX_STEPS):
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=self.messages,
                tools=TOOL_SCHEMAS,
                think=AGENT_THINK,
                options={"temperature": 0},
            )
            message = response.message
            self.messages.append(message)

            if not message.tool_calls:
                reply = _strip_thinking(message.content)
                # Guardrail: the model claims it booked/cancelled, but no such
                # tool succeeded this turn. Tell it so, once, and let it fix that.
                if not nudged and _claims_action(reply) and not self._acted_since(turn_start):
                    nudged = True
                    self._trace("!! reply claims an action that never happened; nudging")
                    self.messages.append({"role": "user", "content": NUDGE})
                    continue
                return reply

            for call in message.tool_calls:
                name = call.function.name
                args = dict(call.function.arguments)
                self._trace(f"-> {name}({json.dumps(args)})")

                if name in SIDE_EFFECT_TOOLS and not self.approve(name, args):
                    result = {
                        "error": "The patient did not approve this action. "
                        "Ask them what they would like to change."
                    }
                else:
                    result = run_tool(name, args)

                self._trace(f"<- {_short(result)}")
                self.tool_log.append(ToolCallRecord(name, args, result))
                self.messages.append(
                    {"role": "tool", "tool_name": name, "content": json.dumps(result)}
                )

        return "Sorry, I couldn't finish that. Please call the clinic on 011 234 5678."

    def _trace(self, line: str) -> None:
        if self.trace:
            self.trace(line)

    def _acted_since(self, start: int) -> bool:
        return any(
            r.name in SIDE_EFFECT_TOOLS and "error" not in r.result
            for r in self.tool_log[start:]
        )


# Phrases that mean "I did it". Deliberately narrow ("is booked", not "book"), so
# offers like "shall I book 3pm?" don't trigger it.
CLAIM_RE = re.compile(
    r"\b(is|has been|have been|are|was|successfully)\s+(booked|confirmed|scheduled|cancell?ed)\b",
    re.I,
)
NUDGE = (
    "[System check] Your last reply says an appointment was booked or cancelled, "
    "but no booking or cancellation tool succeeded in this turn. Either call the "
    "correct tool now, or tell the patient honestly that it has not been done yet."
)


def _claims_action(reply: str) -> bool:
    return bool(CLAIM_RE.search(reply))


def _strip_thinking(text: str) -> str:
    return re.sub(r"^.*?</think>", "", text or "", flags=re.S).strip()


def _short(result: dict, limit: int = 300) -> str:
    text = json.dumps(result)
    return text if len(text) <= limit else text[:limit] + "..."
