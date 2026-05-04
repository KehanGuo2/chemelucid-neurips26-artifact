"""Component 2: Multi-turn agent loop using LLM function calling.

Supports both OpenAI and Anthropic providers. The agent iteratively calls
chemistry tools until it submits an answer or exhausts its budget.

Two modes:
- Stateless (blackboard=None): tool results accumulate in message history.
- Stateful  (blackboard=BlackboardExt): results are distilled into a Blackboard
  which is injected into the system prompt each turn. Message history can be
  truncated while preserving Blackboard context.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from interactive.blackboard_ext import BlackboardExt
from interactive.tool_server import ToolServer, TOOL_DEFINITIONS

# Invalid-submit penalty: budget units deducted for submitting unparseable SMILES
_INVALID_SUBMIT_PENALTY = 5

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parsed LLM response
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    """Unified representation of an LLM response (works for both providers)."""
    text: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    is_submit: bool = False
    submitted_smiles: Optional[str] = None
    raw: Any = None  # Provider-specific raw response

    @property
    def is_tool_call(self) -> bool:
        return len(self.tool_calls) > 0


# ---------------------------------------------------------------------------
# Agent Loop
# ---------------------------------------------------------------------------

class AgentLoop:
    """Multi-turn agent loop for interactive structure elucidation."""

    def __init__(
        self,
        provider: str,
        model: str,
        tool_server: ToolServer,
        blackboard: Optional[BlackboardExt] = None,
        budget: int = 25,
        max_turns: int = 50,
        temperature: float = 0.0,
        reasoning_scaffold: Optional[str] = None,
    ):
        self.provider = provider.lower()  # "openai" or "anthropic"
        self.model = model
        self.tools = tool_server
        self.blackboard = blackboard
        self.budget = budget
        self.max_turns = max_turns
        self.temperature = temperature
        self.reasoning_scaffold = reasoning_scaffold
        self.messages: List[Dict[str, Any]] = []
        self.submitted_smiles: Optional[str] = None
        self._turn_count = 0

        # Lazy-initialised LLM clients (one per AgentLoop instance)
        self._openai_client: Any = None
        self._anthropic_client: Any = None

    def run(self, initial_observation: str) -> Dict[str, Any]:
        """Run the agent loop until submit, budget exhausted, or max turns.

        Args:
            initial_observation: Text describing the unknown molecule's spectra.

        Returns:
            Episode summary dict.
        """
        system_prompt = self._build_system_prompt()

        # Initialize message history
        self.messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": initial_observation},
        ]

        # Inject blackboard context into system prompt if stateful
        if self.blackboard:
            self.messages[0]["content"] = system_prompt + "\n\n" + self.blackboard.render()

        terminated_by = "max_turns"

        while self._turn_count < self.max_turns:
            # Check budget
            if self.tools.get_total_cost() >= self.budget:
                terminated_by = "budget"
                break

            self._turn_count += 1

            # Call LLM
            try:
                response = self._call_llm()
            except Exception as e:
                logger.error(f"LLM call failed on turn {self._turn_count}: {e}")
                terminated_by = "error"
                break

            # Handle response
            if response.is_submit:
                smiles = response.submitted_smiles or ""
                # Validate SMILES before accepting submission
                if not self._is_valid_smiles(smiles):
                    # Invalid SMILES: penalty + continue
                    penalty = _INVALID_SUBMIT_PENALTY
                    self.tools.call_log.append({
                        "tool": "submit_rejected",
                        "args": {"smiles": smiles},
                        "result": {"error": f"Invalid SMILES: '{smiles}'. -{penalty} budget penalty. Fix and retry."},
                        "cost": penalty,
                        "category": "submit",
                    })
                    # Add error to conversation so agent sees it
                    error_msg = (
                        f"Your submission was REJECTED because '{smiles}' is not a valid SMILES string. "
                        f"A penalty of {penalty} budget units has been applied. "
                        f"You have {max(0, self.budget - self.tools.get_total_cost())} units remaining. "
                        f"Please fix the SMILES and try again."
                    )
                    if self.provider in ("openai", "deepinfra", "openrouter"):
                        # For OpenAI, we need to handle the tool call response properly
                        # Find the submit tool call to get its ID
                        submit_tc = None
                        for tc in response.tool_calls:
                            if tc["name"] == "submit":
                                submit_tc = tc
                                break
                        if submit_tc:
                            # Add assistant message with tool call
                            self.messages.append({
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [{
                                    "id": submit_tc.get("id", ""),
                                    "type": "function",
                                    "function": {
                                        "name": "submit",
                                        "arguments": json.dumps({"smiles": smiles}),
                                    },
                                }],
                            })
                            self.messages.append({
                                "role": "tool",
                                "tool_call_id": submit_tc.get("id", ""),
                                "content": error_msg,
                            })
                        else:
                            self.messages.append({"role": "user", "content": error_msg})
                    elif self.provider == "anthropic":
                        submit_tc = None
                        for tc in response.tool_calls:
                            if tc["name"] == "submit":
                                submit_tc = tc
                                break
                        if submit_tc:
                            # Add assistant message with tool_use block
                            content_blocks = []
                            if response.text:
                                content_blocks.append({"type": "text", "text": response.text})
                            content_blocks.append({
                                "type": "tool_use",
                                "id": submit_tc.get("id", ""),
                                "name": "submit",
                                "input": {"smiles": smiles},
                            })
                            self.messages.append({"role": "assistant", "content": content_blocks})
                            self.messages.append({
                                "role": "user",
                                "content": [{
                                    "type": "tool_result",
                                    "tool_use_id": submit_tc.get("id", ""),
                                    "content": error_msg,
                                }],
                            })
                        else:
                            self.messages.append({"role": "user", "content": error_msg})
                    continue  # Don't break — agent gets another chance
                else:
                    # Route submit through ToolServer.execute so the call is
                    # logged in the trajectory (cost=0). dag_grader and the
                    # final-proposal accounting depend on this entry.
                    self.tools.execute("submit", {"smiles": smiles})
                    self.submitted_smiles = smiles
                    terminated_by = "submit"
                    break

            elif response.is_tool_call:
                for tc in response.tool_calls:
                    tool_name = tc["name"]
                    tool_args = tc["args"]

                    # Check budget before each tool call
                    tool_cost = TOOL_DEFINITIONS.get(tool_name, {}).get("cost", 1)
                    if self.tools.get_total_cost() + tool_cost > self.budget and tool_name != "submit":
                        # Inject a budget warning instead
                        self._append_tool_result_to_messages(
                            tc,
                            {"error": f"Budget exceeded. You have {self.budget - self.tools.get_total_cost()} units remaining. Please submit your answer."},
                        )
                        continue

                    # Execute tool
                    result = self.tools.execute(tool_name, tool_args)

                    # Check for submit
                    if tool_name == "submit":
                        smiles = tool_args.get("smiles", "")
                        if not self._is_valid_smiles(smiles):
                            # Invalid SMILES: penalty + continue
                            penalty = _INVALID_SUBMIT_PENALTY
                            self.tools.call_log.append({
                                "tool": "submit_rejected",
                                "args": {"smiles": smiles},
                                "result": {"error": f"Invalid SMILES: '{smiles}'. -{penalty} budget penalty. Fix and retry."},
                                "cost": penalty,
                                "category": "submit",
                            })
                            error_msg = (
                                f"Your submission was REJECTED because '{smiles}' is not a valid SMILES string. "
                                f"A penalty of {penalty} budget units has been applied. "
                                f"You have {max(0, self.budget - self.tools.get_total_cost())} units remaining. "
                                f"Please fix the SMILES and try again."
                            )
                            self._append_tool_result_to_messages(
                                tc, {"error": error_msg}
                            )
                            continue
                        self.submitted_smiles = smiles
                        terminated_by = "submit"
                        break

                    # Update blackboard if stateful
                    if self.blackboard:
                        self.blackboard.add_tool_result(tool_name, tool_args, result)
                        # Re-inject updated blackboard into system prompt
                        self.messages[0]["content"] = (
                            self._build_system_prompt() + "\n\n" + self.blackboard.render()
                        )

                    # Add tool result to conversation
                    self._append_tool_result_to_messages(tc, result)

                if terminated_by == "submit":
                    break

            else:
                # Plain text response (agent reasoning)
                if response.text:
                    self.messages.append({
                        "role": "assistant",
                        "content": response.text,
                    })

                    # Check for SUBMIT: pattern in text
                    submit_match = re.search(
                        r"SUBMIT:\s*(\S+)", response.text, re.IGNORECASE
                    )
                    if submit_match:
                        smiles_candidate = submit_match.group(1).strip()
                        if not self._is_valid_smiles(smiles_candidate):
                            penalty = _INVALID_SUBMIT_PENALTY
                            self.tools.call_log.append({
                                "tool": "submit_rejected",
                                "args": {"smiles": smiles_candidate},
                                "result": {"error": f"Invalid SMILES: '{smiles_candidate}'. -{penalty} budget penalty."},
                                "cost": penalty,
                                "category": "submit",
                            })
                            self.messages.append({
                                "role": "user",
                                "content": (
                                    f"Your submission was REJECTED because '{smiles_candidate}' is not valid SMILES. "
                                    f"-{penalty} budget penalty. "
                                    f"{max(0, self.budget - self.tools.get_total_cost())} units remaining. Fix and retry."
                                ),
                            })
                        else:
                            # Route through ToolServer.execute so the submit
                            # is logged to the trajectory (Bug 1).
                            self.tools.execute("submit", {"smiles": smiles_candidate})
                            self.submitted_smiles = smiles_candidate
                            terminated_by = "submit"
                            break

        return {
            "submitted_smiles": self.submitted_smiles,
            "trajectory": self.tools.get_trajectory(),
            "total_cost": self.tools.get_total_cost(),
            "budget_remaining": max(0, self.budget - self.tools.get_total_cost()),
            "num_turns": self._turn_count,
            "terminated_by": terminated_by,
        }

    # ------------------------------------------------------------------
    # SMILES validation (for invalid-submit penalty)
    # ------------------------------------------------------------------

    @staticmethod
    def _is_valid_smiles(smiles: str) -> bool:
        """Return True if RDKit can parse the SMILES string."""
        if not smiles or not smiles.strip():
            return False
        try:
            from rdkit import Chem
            return Chem.MolFromSmiles(smiles) is not None
        except ImportError:
            # Without RDKit, accept anything non-empty
            return bool(smiles.strip())

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        remaining = self.budget - self.tools.get_total_cost()

        # Build the advertised tool list dynamically so it matches the schema
        # actually exposed to the model (e.g. get_full_spectrum is hidden in
        # partial info_mode; HNMR-only tools are hidden in CNMR mode).
        # Map presentation sections -> tool categories from TOOL_DEFINITIONS.
        # "Structure Analysis" bundles structure_analysis + prediction +
        # validation tools as in the original prompt.
        _SECTIONS = [
            ("Spectrum Query", ("spectrum_query",)),
            ("Computation", ("computation",)),
            ("Structure Analysis", ("structure_analysis", "prediction", "validation")),
            ("Submit", ("submit",)),
        ]
        _TOOL_BLURB = {
            "query_spectrum": 'query_spectrum: Query NMR peaks in a specific ppm range (nucleus="13C" or "1H", ppm_min, ppm_max)',
            "get_full_spectrum": 'get_full_spectrum: Get the complete NMR spectrum (nucleus="13C" or "1H")',
            "get_multiplicity": "get_multiplicity: Get multiplicity (s/d/t/q/m/dd/dt...) for 1H NMR peaks (optional ppm range)",
            "get_integration": "get_integration: Get integration (number of protons) for 1H NMR peaks (optional ppm range)",
            "get_coupling": "get_coupling: Get J-coupling values (Hz) for 1H NMR peaks (optional ppm range)",
            "compute_unsaturation": "compute_unsaturation: Calculate degree of unsaturation (DBE) from molecular formula",
            "compute_molecular_weight": "compute_molecular_weight: Calculate MW from formula",
            "substructure_check": "substructure_check: Check if a SMILES contains a SMARTS substructure",
            "count_carbons": "count_carbons: Count carbons in a SMILES structure",
            "check_symmetry": "check_symmetry: Analyze symmetry / unique carbon environments",
            "validate_smiles": "validate_smiles: Validate SMILES and get canonical form + formula",
            "predict_nmr": "predict_nmr: Predict approximate NMR shifts for a candidate structure",
            "compare_spectra": "compare_spectra: Compare predicted vs observed spectra quantitatively",
            "submit": "submit: Submit your final answer (SMILES string)",
        }
        available = set(self.tools.get_available_tools())
        sections: List[str] = []
        for header, cats in _SECTIONS:
            lines = [
                f"- {_TOOL_BLURB[name]}"
                for name, defn in TOOL_DEFINITIONS.items()
                if defn.get("category") in cats and name in available and name in _TOOL_BLURB
            ]
            if lines:
                sections.append(f"**{header}:**\n" + "\n".join(lines))
        tools_block = "\n\n".join(sections)

        base = f"""You are a chemistry research agent tasked with identifying an unknown molecular structure from NMR spectroscopy data.

You have access to the following tools:

{tools_block}

STRATEGY:
1. Start by computing the degree of unsaturation from the molecular formula
2. Query specific spectral regions to identify functional groups:
   - 13C: 110-160 ppm (aromatic), 160-220 ppm (carbonyl), 0-50 ppm (alkyl), 50-100 ppm (C-O/C-N)
   - 1H: 6.5-8.0 ppm (aromatic), 0.25-2.5 ppm (alkyl), 3.0-4.5 ppm (hetero C-H), 8.5-10 ppm (aldehyde)
3. For 1H NMR: use get_multiplicity, get_integration, get_coupling to determine splitting patterns
4. Use substructure_check and check_symmetry to test structural hypotheses
5. When you have a candidate, use predict_nmr + compare_spectra to validate
6. Use validate_smiles to ensure your candidate has the correct formula
7. When confident, call submit(smiles=...) with your answer

BUDGET: {remaining} tool-call units remaining (out of {self.budget}).
Be efficient — each tool call costs units. Prioritize high-information tools first.
When budget is low, submit your best guess rather than wasting calls."""

        if self.reasoning_scaffold:
            return base + "\n\n" + self.reasoning_scaffold
        return base

    # ------------------------------------------------------------------
    # LLM calling (provider-specific)
    # ------------------------------------------------------------------

    def _call_llm(self) -> LLMResponse:
        """Dispatch to the appropriate provider."""
        if self.provider in ("openai", "deepinfra", "openrouter"):
            return self._call_openai()
        elif self.provider == "anthropic":
            return self._call_anthropic()
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _get_openai_client(self):
        """Return cached OpenAI-compatible client (created on first call)."""
        if self._openai_client is None:
            from openai import OpenAI
            import os

            if self.provider == "deepinfra":
                self._openai_client = OpenAI(
                    api_key=os.getenv("DEEPINFRA_API_KEY", ""),
                    base_url="https://api.deepinfra.com/v1/openai",
                )
            elif self.provider == "openrouter":
                self._openai_client = OpenAI(
                    api_key=os.getenv("OPENROUTER_API_KEY", ""),
                    base_url="https://openrouter.ai/api/v1",
                )
            else:
                self._openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
        return self._openai_client

    def _get_anthropic_client(self):
        """Return cached Anthropic client (created on first call)."""
        if self._anthropic_client is None:
            import anthropic
            import os

            self._anthropic_client = anthropic.Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            )
        return self._anthropic_client

    def _is_reasoning_model(self) -> bool:
        m = (self.model or "").lower()
        return m.startswith(("o1", "o3", "o4", "gpt-5"))

    def _call_with_retry(self, fn, max_retries: int = 6):
        """Retry on 429 / transient 5xx with exponential backoff + jitter."""
        import random
        import time as _time
        delay = 2.0
        last_exc = None
        for attempt in range(max_retries):
            try:
                return fn()
            except Exception as e:
                last_exc = e
                msg = str(e)
                retryable = ("429" in msg or "rate_limit" in msg.lower()
                             or "overloaded" in msg.lower()
                             or "503" in msg or "502" in msg or "504" in msg)
                if not retryable or attempt == max_retries - 1:
                    raise
                sleep_s = delay + random.uniform(0, 1.0)
                logger.warning(f"Retryable API error (attempt {attempt+1}/{max_retries}, sleeping {sleep_s:.1f}s): {msg[:200]}")
                _time.sleep(sleep_s)
                delay = min(delay * 2, 30.0)
        raise last_exc  # unreachable, satisfies linters

    def _call_openai(self) -> LLMResponse:
        """Call OpenAI-compatible API with function calling."""
        client = self._get_openai_client()

        tools = self.tools.get_tool_definitions_for_llm()

        # Filter system messages for models that don't support them
        messages = self.messages

        kwargs = dict(model=self.model, messages=messages, tools=tools, tool_choice="auto")
        if not self._is_reasoning_model():
            kwargs["temperature"] = self.temperature

        response = self._call_with_retry(lambda: client.chat.completions.create(**kwargs))

        choice = response.choices[0]
        msg = choice.message

        # Parse tool calls
        if msg.tool_calls:
            tool_calls = []
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "args": args,
                })

            # Check if any tool call is "submit"
            for tc in tool_calls:
                if tc["name"] == "submit":
                    return LLMResponse(
                        tool_calls=tool_calls,
                        is_submit=True,
                        submitted_smiles=tc["args"].get("smiles", ""),
                        raw=msg,
                    )

            # Store the raw assistant message for conversation continuity
            self.messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            })

            return LLMResponse(tool_calls=tool_calls, raw=msg)

        # Plain text
        text = msg.content or ""
        return LLMResponse(text=text, raw=msg)

    def _call_anthropic(self) -> LLMResponse:
        """Call Anthropic API with tool_use."""
        client = self._get_anthropic_client()

        tools = self.tools.get_tool_definitions_for_anthropic()

        # Extract system message and filter to user/assistant messages
        system_text = ""
        api_messages = []
        for m in self.messages:
            if m["role"] == "system":
                system_text += m["content"] + "\n"
            else:
                api_messages.append(m)

        if not api_messages:
            api_messages.append({"role": "user", "content": "Begin analysis."})

        anthropic_kwargs = dict(
            model=self.model,
            max_tokens=4096,
            system=system_text.strip(),
            messages=api_messages,
            tools=tools,
        )
        if "claude-opus-4-7" not in self.model and "claude-sonnet-4-6" not in self.model:
            anthropic_kwargs["temperature"] = self.temperature
        response = self._call_with_retry(lambda: client.messages.create(**anthropic_kwargs))

        # Parse Anthropic response blocks
        tool_calls = []
        text_parts = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "args": block.input,
                })

        text = "\n".join(text_parts) if text_parts else None

        if tool_calls:
            # Check for submit
            for tc in tool_calls:
                if tc["name"] == "submit":
                    return LLMResponse(
                        text=text,
                        tool_calls=tool_calls,
                        is_submit=True,
                        submitted_smiles=tc["args"].get("smiles", ""),
                        raw=response,
                    )

            # Store assistant message with tool_use blocks
            content_blocks = []
            if text:
                content_blocks.append({"type": "text", "text": text})
            for tc in tool_calls:
                content_blocks.append({
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["name"],
                    "input": tc["args"],
                })

            self.messages.append({
                "role": "assistant",
                "content": content_blocks,
            })

            return LLMResponse(text=text, tool_calls=tool_calls, raw=response)

        return LLMResponse(text=text, raw=response)

    # ------------------------------------------------------------------
    # Message helpers
    # ------------------------------------------------------------------

    def _append_tool_result_to_messages(
        self,
        tool_call: Dict[str, Any],
        result: Dict[str, Any],
    ):
        """Append tool result to message history (provider-agnostic)."""
        result_text = json.dumps(result, default=str)

        if self.provider in ("openai", "deepinfra", "openrouter"):
            self.messages.append({
                "role": "tool",
                "tool_call_id": tool_call.get("id", ""),
                "content": result_text,
            })
        elif self.provider == "anthropic":
            self.messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_call.get("id", ""),
                        "content": result_text,
                    }
                ],
            })
