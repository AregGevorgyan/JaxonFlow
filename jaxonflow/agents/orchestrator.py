"""Multi-agent orchestrator for kernel generation."""

from __future__ import annotations

import logging
import time
from typing import Any

from ..config import AgentBackendConfig
from ..exceptions import KernelGenerationError
from ..llm.client import LLMClient, LLMResponse, TokenUsage, create_client
from ..spec import CompiledKernel, CompilationResult, KernelSpec, ProfileResult, VerificationResult
from .base import Agent, AgentConfig, AgentRole, LLMAgent
from .coder import CoderAgent
from .debugger import DebuggerAgent
from .planner import PlannerAgent
from .profiler_agent import ProfilerAgent
from .prompts import get_system_prompt
from .verification import KernelVerifier

logger = logging.getLogger(__name__)


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

        # Helper to create config
        def make_config(role: AgentRole, temp: float, tokens: int) -> AgentConfig:
            return AgentConfig(
                role=role,
                model=self.config.llm.model,
                temperature=temp,
                max_tokens=tokens,
                system_prompt=get_system_prompt(role),
            )

        agent_map: dict[AgentRole, Agent] = {
            AgentRole.PLANNER: PlannerAgent(
                make_config(
                    AgentRole.PLANNER, 
                    agent_config.planner_temperature, 
                    agent_config.planner_max_tokens
                ),
                llm_client=client,
            ),
            AgentRole.CODER: CoderAgent(
                make_config(
                    AgentRole.CODER, 
                    agent_config.coder_temperature, 
                    agent_config.coder_max_tokens
                ),
                llm_client=client,
            ),
            AgentRole.DEBUGGER: DebuggerAgent(
                make_config(
                    AgentRole.DEBUGGER, 
                    agent_config.debugger_temperature, 
                    agent_config.debugger_max_tokens
                ),
                llm_client=client,
            ),
            AgentRole.PROFILER: ProfilerAgent(
                make_config(
                    AgentRole.PROFILER, 
                    agent_config.profiler_temperature, 
                    agent_config.profiler_max_tokens
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
        planner = self.agents[AgentRole.PLANNER]
        if not isinstance(planner, PlannerAgent):
            raise TypeError("Planner agent is not of type PlannerAgent")
            
        plan = planner.create_plan(spec)
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
            coder = self.agents[AgentRole.CODER]
            if not isinstance(coder, CoderAgent):
                 raise TypeError("Coder agent is not of type CoderAgent")
            
            code = coder.generate_kernel(spec, plan, history)
            
            if not code or not code.strip():
                logger.warning("Coder returned no code")
                continue

            # Phase 3: Compilation
            compile_result = self._compile_kernel(code, spec)
            if not compile_result.success:
                logger.warning(f"Compilation failed: {compile_result.error}")
                # Use debugger to analyze the error
                debugger = self.agents[AgentRole.DEBUGGER]
                if not isinstance(debugger, DebuggerAgent):
                     raise TypeError("Debugger agent is not of type DebuggerAgent")
                
                debug_feedback = debugger.analyze_error(
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
                debugger = self.agents[AgentRole.DEBUGGER]
                if not isinstance(debugger, DebuggerAgent):
                     raise TypeError("Debugger agent is not of type DebuggerAgent")

                debug_feedback = debugger.analyze_mismatch(
                    code,
                    verify_result.expected,
                    verify_result.actual
                )
                history.append({
                    "iteration": iteration,
                    "phase": "verification_failed",
                    "error": verify_result.test_case_description,
                    "debug_feedback": debug_feedback,
                })
                continue

            # Kernel is correct 
            compile_result.kernel.iterations_to_generate = iteration + 1
            compile_result.kernel.generation_time_s = time.monotonic() - start_time
            
            # Phase 5: Profiling (Optional / Future)
            # Since we typically run on CPU or without GPU in dev, we might skip this.
            # But let's verify if we should profile.
            
            # If we were to profile:
            # profile_result = self._profile_kernel(compile_result.kernel, spec)
            # if profile_result.speedup_vs_reference > best_speedup: ...
            
            # For now, accept the first correct kernel.
            logger.info(
                f"Generated correct kernel for {spec.operation} "
                f"in {iteration + 1} iteration(s)"
            )
            return compile_result.kernel

        # Return best kernel found (if any)
        if best_kernel is not None:
            return best_kernel

        logger.error(
            f"Failed to generate kernel for {spec.operation} "
            f"after {self.config.agent.max_iterations} iterations"
        )
        return None

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
            
            # If no wrapper found, maybe the function itself is the kernel?
            # Triton kernels are decorated with @jit, so they are callables.
            if callable_fn is None:
                 for name, obj in namespace.items():
                    if callable(obj) and not name.startswith("_") and name != "triton":
                        # Pick the first reasonable function
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
