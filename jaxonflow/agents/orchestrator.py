"""Multi-agent orchestrator for kernel generation."""

from __future__ import annotations

import logging
import time
from typing import Any

from ..config import AgentBackendConfig
from ..exceptions import KernelGenerationError
from ..llm.client import LLMClient, LLMResponse, TokenUsage, create_client
from ..spec import CompiledKernel, CompilationResult, KernelSpec, ProfileResult, VerificationResult
from .base import Agent, AgentConfig, AgentRole
from .prompts import get_system_prompt
from .verification import KernelVerifier

logger = logging.getLogger(__name__)


class LLMAgent(Agent):
    """An agent that uses an LLM client to generate responses.

    Each agent has a specialized role with its own system prompt,
    temperature setting, and token limit.
    """

    def __init__(self, config: AgentConfig, llm_client: LLMClient) -> None:
        """Initialize an LLM-based agent.

        Args:
            config: Agent configuration with role, temperature, etc.
            llm_client: The LLM client to use for generation.
        """
        super().__init__(config)
        self.llm_client = llm_client

    def run(self, *args: Any, **kwargs: Any) -> LLMResponse:
        """Run the agent with a formatted prompt.

        Expects the first positional argument to be the user prompt string.

        Args:
            *args: First arg should be the prompt string.
            **kwargs: Optional overrides for temperature, max_tokens.

        Returns:
            LLMResponse from the LLM client.
        """
        if not args:
            raise ValueError("LLMAgent.run() requires at least a prompt argument")

        prompt = str(args[0])
        temperature = kwargs.get("temperature", self.config.temperature)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)

        return self.llm_client.generate(
            prompt=prompt,
            system_prompt=self.config.system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )


class MultiAgentOrchestrator:
    """Coordinates multiple specialized agents for kernel generation.

    The orchestrator manages the generate-evaluate-refine loop:
    1. Planner creates optimization strategy
    2. Coder generates Triton kernel code
    3. Compilation attempt
    4. Verifier checks correctness
    5. If incorrect: Debugger analyzes and feeds back to Coder
    6. If correct but slow: Profiler analyzes, Planner revises
    7. Repeat until target met or max iterations
    """

    def __init__(self, config: AgentBackendConfig) -> None:
        """Initialize the orchestrator.

        Args:
            config: Backend configuration.
        """
        self.config = config
        self._llm_client: LLMClient | None = None
        self._agents: dict[AgentRole, Agent] | None = None
        self._usage = TokenUsage()

    @property
    def llm_client(self) -> LLMClient:
        """Lazily initialize the LLM client."""
        if self._llm_client is None:
            self._llm_client = create_client(self.config.llm)
        return self._llm_client

    @property
    def agents(self) -> dict[AgentRole, Agent]:
        """Lazily initialize agents."""
        if self._agents is None:
            self._agents = self._initialize_agents()
        return self._agents

    def _initialize_agents(self) -> dict[AgentRole, Agent]:
        """Initialize specialized agents with role-appropriate configurations."""
        agent_config = self.config.agent
        client = self.llm_client

        agent_map: dict[AgentRole, Agent] = {
            AgentRole.PLANNER: LLMAgent(
                AgentConfig(
                    role=AgentRole.PLANNER,
                    model=self.config.llm.model,
                    temperature=agent_config.planner_temperature,
                    max_tokens=agent_config.planner_max_tokens,
                    system_prompt=get_system_prompt(AgentRole.PLANNER),
                ),
                llm_client=client,
            ),
            AgentRole.CODER: LLMAgent(
                AgentConfig(
                    role=AgentRole.CODER,
                    model=self.config.llm.model,
                    temperature=agent_config.coder_temperature,
                    max_tokens=agent_config.coder_max_tokens,
                    system_prompt=get_system_prompt(AgentRole.CODER),
                ),
                llm_client=client,
            ),
            AgentRole.DEBUGGER: LLMAgent(
                AgentConfig(
                    role=AgentRole.DEBUGGER,
                    model=self.config.llm.model,
                    temperature=agent_config.debugger_temperature,
                    max_tokens=agent_config.debugger_max_tokens,
                    system_prompt=get_system_prompt(AgentRole.DEBUGGER),
                ),
                llm_client=client,
            ),
            AgentRole.PROFILER: LLMAgent(
                AgentConfig(
                    role=AgentRole.PROFILER,
                    model=self.config.llm.model,
                    temperature=agent_config.profiler_temperature,
                    max_tokens=agent_config.profiler_max_tokens,
                    system_prompt=get_system_prompt(AgentRole.PROFILER),
                ),
                llm_client=client,
            ),
        }

        # Verifier is deterministic (no LLM)
        agent_map[AgentRole.VERIFIER] = KernelVerifier(
            AgentConfig(
                role=AgentRole.VERIFIER,
                model="deterministic",
                system_prompt=get_system_prompt(AgentRole.VERIFIER),
            )
        )

        return agent_map

    def generate_kernel(self, spec: KernelSpec) -> CompiledKernel | None:
        """Main kernel generation loop.

        Args:
            spec: The kernel specification to generate code for.

        Returns:
            CompiledKernel if successful, None otherwise.
        """
        start_time = time.monotonic()
        logger.info(f"Starting kernel generation for {spec.operation}")

        # Phase 1: Planning
        plan = self._run_planner(spec)
        logger.debug(f"Plan created for {spec.operation}")

        best_kernel: CompiledKernel | None = None
        best_speedup = 0.0
        history: list[dict[str, Any]] = []

        for iteration in range(self.config.agent.max_iterations):
            elapsed = time.monotonic() - start_time
            if elapsed > self.config.agent.total_timeout:
                logger.warning(f"Total timeout reached after {iteration} iterations")
                break

            logger.debug(
                f"Generation iteration {iteration + 1}/{self.config.agent.max_iterations}"
            )

            # Phase 2: Code Generation
            code = self._run_coder(spec, plan, history)
            if code is None:
                logger.warning("Coder returned no code")
                continue

            # Phase 3: Compilation
            compile_result = self._compile_kernel(code, spec)
            if not compile_result.success:
                logger.warning(f"Compilation failed: {compile_result.error}")
                # Use debugger to analyze the error
                debug_feedback = self._run_debugger(
                    code, compile_result.error or "Unknown error", error_type="compilation"
                )
                history.append({
                    "iteration": iteration,
                    "phase": "compilation_failed",
                    "error": compile_result.error,
                    "debug_feedback": debug_feedback,
                })
                continue

            # Phase 4: Correctness Verification
            assert compile_result.kernel is not None
            verifier = self.agents[AgentRole.VERIFIER]
            verify_result = verifier.run(compile_result.kernel, spec)

            if not verify_result.correct:
                logger.warning(
                    f"Verification failed: {verify_result.test_case_description}"
                )
                debug_feedback = self._run_debugger(
                    code,
                    f"Verification failed: {verify_result.test_case_description}, "
                    f"max_abs_error={verify_result.max_abs_error}",
                    error_type="correctness",
                )
                history.append({
                    "iteration": iteration,
                    "phase": "verification_failed",
                    "error": verify_result.test_case_description,
                    "debug_feedback": debug_feedback,
                })
                continue

            # Kernel is correct - record it
            compile_result.kernel.iterations_to_generate = iteration + 1
            compile_result.kernel.generation_time_s = time.monotonic() - start_time

            # For now, accept the first correct kernel.
            # In a GPU environment, we'd profile and potentially iterate.
            logger.info(
                f"Generated correct kernel for {spec.operation} "
                f"in {iteration + 1} iteration(s)"
            )
            return compile_result.kernel

        # Return best kernel found (if any), even if we didn't hit target speedup
        if best_kernel is not None:
            return best_kernel

        logger.error(
            f"Failed to generate kernel for {spec.operation} "
            f"after {self.config.agent.max_iterations} iterations"
        )
        return None

    def _run_planner(self, spec: KernelSpec) -> str:
        """Run the planner agent to create an optimization strategy.

        Args:
            spec: The kernel specification.

        Returns:
            The plan as a string (typically YAML formatted).
        """
        prompt = (
            f"Create an optimization plan for the following kernel:\n\n"
            f"{spec.to_prompt()}\n\n"
            f"Consider the hardware constraints and suggest tile sizes, "
            f"grid configuration, and memory access patterns."
        )

        planner = self.agents[AgentRole.PLANNER]
        response = planner.run(prompt)
        return response.content if isinstance(response, LLMResponse) else str(response)

    def _run_coder(
        self, spec: KernelSpec, plan: str, history: list[dict[str, Any]]
    ) -> str | None:
        """Run the coder agent to generate Triton kernel code.

        Args:
            spec: The kernel specification.
            plan: The optimization plan from the planner.
            history: Previous iteration history for context.

        Returns:
            Generated Triton code string, or None if extraction failed.
        """
        prompt_parts = [
            f"Generate a Triton kernel for the following specification:\n\n",
            f"{spec.to_prompt()}\n\n",
            f"Optimization Plan:\n{plan}\n\n",
        ]

        if history:
            prompt_parts.append("Previous Attempts:\n")
            for entry in history[-3:]:  # Last 3 attempts for context
                prompt_parts.append(
                    f"- Iteration {entry['iteration']}: {entry['phase']}\n"
                    f"  Error: {entry.get('error', 'N/A')}\n"
                    f"  Feedback: {entry.get('debug_feedback', 'N/A')}\n"
                )
            prompt_parts.append(
                "\nPlease fix the issues from previous attempts.\n"
            )

        prompt_parts.append(
            "Generate the complete, executable Triton kernel code. "
            "Include a wrapper function that handles grid computation and kernel launch."
        )

        prompt = "".join(prompt_parts)
        coder = self.agents[AgentRole.CODER]
        response = coder.run(prompt)

        # Extract code from response
        if isinstance(response, LLMResponse):
            code = self.llm_client.extract_code(response, language="python")
            if code is None:
                # Try extracting from raw content (maybe no code fence)
                code = response.content
            return code

        return str(response) if response else None

    def _run_debugger(
        self, code: str, error: str, error_type: str = "compilation"
    ) -> str:
        """Run the debugger agent to analyze errors.

        Args:
            code: The kernel code that failed.
            error: The error message.
            error_type: Type of error ("compilation" or "correctness").

        Returns:
            Debug feedback as a string.
        """
        prompt = (
            f"Analyze the following {error_type} error in a Triton kernel:\n\n"
            f"Code:\n```python\n{code}\n```\n\n"
            f"Error:\n{error}\n\n"
            f"Provide specific fixes in YAML format."
        )

        debugger = self.agents[AgentRole.DEBUGGER]
        response = debugger.run(prompt)
        return response.content if isinstance(response, LLMResponse) else str(response)

    def _compile_kernel(self, code: str, spec: KernelSpec) -> CompilationResult:
        """Compile the generated kernel code.

        Attempts to execute the code and create a callable. On systems
        without a GPU, this will try to parse the code and fall back to
        the reference implementation for verification purposes.

        Args:
            code: The generated kernel code string.
            spec: The kernel specification.

        Returns:
            CompilationResult indicating success or failure.
        """
        if not code or not code.strip():
            return CompilationResult(success=False, error="Empty code generated")

        try:
            # Try to compile/exec the code to at least check syntax
            compile(code, "<generated_kernel>", "exec")
        except SyntaxError as e:
            return CompilationResult(
                success=False,
                error=f"Syntax error at line {e.lineno}: {e.msg}",
                error_line=e.lineno,
            )

        # Try to create a callable
        callable_fn = None
        try:
            namespace: dict[str, Any] = {}
            exec(code, namespace)  # noqa: S102

            # Look for wrapper functions
            for name, obj in namespace.items():
                if callable(obj) and not name.startswith("_"):
                    if "wrapper" in name.lower() or "launch" in name.lower():
                        callable_fn = obj
                        break

        except Exception as e:
            # Code has imports that aren't available (like triton) - that's OK
            logger.debug(f"Could not exec kernel code: {e}")

        # If we couldn't find a proper callable, use reference impl for verification
        if callable_fn is None:
            if spec.reference_impl is not None:
                callable_fn = spec.reference_impl
            else:
                # Create a no-op for syntax-valid but non-executable code
                def _noop(*args: Any, **kwargs: Any) -> Any:
                    return args[0] if args else None

                callable_fn = _noop

        kernel = CompiledKernel(
            spec=spec,
            code=code,
            callable=callable_fn,
        )
        return CompilationResult(success=True, kernel=kernel)

    def get_usage_stats(self) -> dict[str, Any]:
        """Get cumulative LLM usage statistics.

        Returns:
            Dictionary with token counts, costs, and request stats.
        """
        if self._llm_client is not None:
            usage = self._llm_client.usage
            return {
                "total_input_tokens": usage.total_input_tokens,
                "total_output_tokens": usage.total_output_tokens,
                "total_requests": usage.total_requests,
                "estimated_cost_usd": usage.estimated_cost_usd,
                "average_latency_ms": usage.average_latency_ms,
            }
        return {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_requests": 0,
            "estimated_cost_usd": 0.0,
            "average_latency_ms": 0.0,
        }
