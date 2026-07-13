# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Iterable
from contextlib import AsyncExitStack, ExitStack
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, TypeAlias, overload

from fast_depends import Provider
from fast_depends.core import CallModel
from fast_depends.pydantic.schema import get_schema

from autogen.beta.annotations import Context
from autogen.beta.events import ToolCallEvent, ToolErrorEvent, ToolResultEvent
from autogen.beta.middleware import BaseMiddleware, ToolExecution, ToolMiddleware, ToolResultType
from autogen.beta.tools.example_discovery import Proposer, play_with_tool
from autogen.beta.tools.schemas import ToolSchema
from autogen.beta.tools.tool import Tool
from autogen.beta.tools.tool_examples import enhance_tool_description
from autogen.beta.utils import CONTEXT_OPTION_NAME, build_model

FunctionParameters: TypeAlias = dict[str, Any]


@dataclass(slots=True)
class FunctionDefinition:
    name: str
    description: str = ""
    parameters: FunctionParameters = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.parameters.pop("title", None)


@dataclass(slots=True)
class FunctionToolSchema(ToolSchema):
    type: str = field(default="function", init=False)
    function: FunctionDefinition = field(default_factory=lambda: FunctionDefinition(name=""))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FunctionToolSchema":
        func_data = data.get("function", {})
        return cls(function=FunctionDefinition(**func_data))


class FunctionTool(Tool):
    __slots__ = (
        "model",
        "name",
        "schema",
        "provider",
        "_middleware",
        "_func",
    )

    def __init__(
        self,
        model: CallModel,
        *,
        name: str,
        description: str,
        schema: FunctionParameters,
        middleware: Iterable[ToolMiddleware] = (),
        func: Callable[..., Any] | None = None,
    ) -> None:
        self.model = model
        self._middleware: tuple[ToolMiddleware, ...] = tuple(middleware)
        # The raw callable, retained so the tool can be *played* with at
        # runtime to discover verified usage examples (see discover_examples).
        self._func = func

        self.schema = FunctionToolSchema(
            function=FunctionDefinition(
                name=name,
                description=description,
                parameters=schema,
            )
        )

        self.provider: Provider | None = None
        self.name = name

    def with_middleware(self, *middleware: ToolMiddleware) -> "FunctionTool":
        """Return a new FunctionTool with additional middleware appended.

        Does not modify the original tool.
        """
        cloned = deepcopy(self)
        cloned._middleware = tuple(middleware) + self._middleware
        return cloned

    async def schemas(self, context: "Context") -> list[FunctionToolSchema]:
        return [self.schema]

    async def discover_examples(
        self,
        *,
        proposer: "Proposer | None" = None,
        max_iterations: int = 3,
        expand_num: int = 4,
        top_k: int = 3,
    ) -> "FunctionTool":
        """Play with this tool to discover verified usage examples.

        Executes the tool with proposed invocations, keeps the ones that
        succeed, and returns a new ``FunctionTool`` whose description embeds
        those execution-grounded examples. Adapted from PLAY2PROMPT
        (https://arxiv.org/abs/2503.14432v2). Returns ``self`` unchanged when
        the underlying callable is unavailable.
        """
        if self._func is None:
            return self

        fn = self.schema.function
        result = await play_with_tool(
            self._func,
            name=fn.name,
            description=fn.description,
            parameters=fn.parameters,
            proposer=proposer,
            max_iterations=max_iterations,
            expand_num=expand_num,
            top_k=top_k,
        )
        return FunctionTool(
            self.model,
            name=fn.name,
            description=result.enhanced_description,
            schema=fn.parameters,
            middleware=self._middleware,
            func=self._func,
        )

    def set_provider(self, provider: Provider) -> None:
        self.provider = provider

    @staticmethod
    def ensure_tool(
        func: "Tool | Callable[..., Any]",
        *,
        provider: Provider | None = None,
    ) -> "Tool":
        t = deepcopy(func) if isinstance(func, Tool) else tool(func)
        t.set_provider(provider)
        return t

    def register(
        self,
        stack: "ExitStack | AsyncExitStack",
        context: "Context",
        *,
        middleware: Iterable["BaseMiddleware"] = (),
    ) -> None:
        execution: ToolExecution = self
        for hook in reversed(self._middleware):
            execution = _wrap_middleware(hook, execution)
        for mw in middleware:
            execution = _wrap_middleware(mw.on_tool_execution, execution)

        async def execute(event: "ToolCallEvent", context: "Context") -> None:
            result = await execution(event, context)
            await context.send(result)

        stack.enter_context(context.stream.where(ToolCallEvent.name == self.schema.function.name).sub_scope(execute))

    async def __call__(self, event: "ToolCallEvent", context: "Context") -> "ToolResultEvent":
        try:
            async with AsyncExitStack() as stack:
                result = await self.model.asolve(
                    **(event.serialized_arguments | {CONTEXT_OPTION_NAME: context}),
                    stack=stack,
                    cache_dependencies={},
                    dependency_provider=self.provider,
                )

            return ToolResultEvent.from_call(event, result=result)

        except Exception as e:
            return ToolErrorEvent.from_call(event, error=e)


@overload
def tool(
    function: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
    schema: FunctionParameters | None = None,
    sync_to_thread: bool = True,
    add_examples: bool = False,
    middleware: Iterable[ToolMiddleware] = (),
) -> FunctionTool: ...


@overload
def tool(
    function: None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    schema: FunctionParameters | None = None,
    sync_to_thread: bool = True,
    add_examples: bool = False,
    middleware: Iterable[ToolMiddleware] = (),
) -> Callable[[Callable[..., Any]], FunctionTool]: ...


def tool(
    function: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    schema: FunctionParameters | None = None,
    sync_to_thread: bool = True,
    add_examples: bool = False,
    middleware: Iterable[ToolMiddleware] = (),
) -> FunctionTool | Callable[[Callable[..., Any]], FunctionTool]:
    def make_tool(f: Callable[..., Any]) -> FunctionTool:
        call_model = build_model(
            f,
            sync_to_thread=sync_to_thread,
            serialize_result=False,
        )

        base_schema = schema or get_schema(
            call_model,
            exclude=(CONTEXT_OPTION_NAME,),
        )
        tool_name = name or f.__name__
        base_description = description or f.__doc__ or ""

        # Enhance description with examples if requested
        final_description = (
            enhance_tool_description(tool_name, base_description, base_schema) if add_examples else base_description
        )

        return FunctionTool(
            call_model,
            name=tool_name,
            description=final_description,
            schema=base_schema,
            middleware=middleware,
            func=f,
        )

    if function:
        return make_tool(function)
    return make_tool


def _wrap_middleware(hook: "ToolMiddleware", inner: "ToolExecution") -> "ToolExecution":
    async def call(event: "ToolCallEvent", context: "Context") -> "ToolResultType":
        return await hook(inner, event, context)

    return call
