from __future__ import annotations

import dataclasses
import decimal
import itertools
from collections import ChainMap
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Iterable,
        Iterator,
        MutableMapping,
        Sequence,
    )
    from typing import (
        Any,
        ClassVar,
        Self,
        TypeVar,
    )

from . import dbg, expr, tree, utils
from .parse import (
    ParsedToken,
    ParsedTokenRole,
    ParsedTokenTag,
    ParseError,
    RestartParser,
    basic_parser,
    parser_decorator,
    skipspace,
    yieldspace,
)
from .token import TokenError
from .tree import KeyValues
from .utils import CaseInsensitiveDict

if TYPE_CHECKING:
    from .parse import ParserFn, ParserIO
    from .token import AnyToken


def read_balanced(
    tokens: Iterable[ParsedToken],
    depth: int = 0,
) -> Iterator[ParsedToken]:
    tokens = iter(tokens)

    # Yield leading whitespace and comments.
    nonspace = yield from yieldspace(tokens, depth)

    # Yield tokens until we reach zero depth.
    for token in itertools.chain([nonspace], tokens):
        match token.tag:
            case ParsedTokenTag.EOF:
                errmsg = "missing }"
                raise ParseError(errmsg, token)

            # Braces are not considered to be inside of their sections.

            case ParsedTokenTag.PLAIN if token.data == "{":
                if depth > 0:
                    token.meta["depth"] = depth
                    yield token

                depth += 1

            case ParsedTokenTag.PLAIN if token.data == "}":
                depth -= 1

                if depth > 0:
                    token.meta["depth"] = depth
                    yield token
                else:
                    break

            case _:
                token.meta["depth"] = depth
                yield token

                if depth == 0:
                    break


class PreprocessorError(TokenError):
    pass


class DirectiveError(Exception):
    def __init__(self, message: str, token: ParsedToken | None = None):
        self.message = message
        self.token = token


@dataclass(slots=True)
class Definition:
    params: list[ParsedToken]
    body: list[ParsedToken]
    arity: int = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.arity = len(self.params)


if TYPE_CHECKING:
    T = TypeVar("T")
    DirectiveFn = Callable[
        [T, list[ParsedToken], Iterator[ParsedToken]],
        Iterable[ParsedToken] | None,
    ]


class DirectiveHandler(NamedTuple):
    names: list[str]
    handler: DirectiveFn[Any]


class Directives:
    _directives: ClassVar[dict[str, DirectiveFn[Self]]] = {}

    @staticmethod
    def directive(
        *names: str,
    ) -> Callable[[DirectiveFn[T]], DirectiveHandler]:
        def dectorator(fn: DirectiveFn[T]) -> DirectiveHandler:
            return DirectiveHandler(list(names), fn)

        return dectorator

    def __init_subclass__(cls) -> None:
        attrs = vars(cls)

        if "_directives" not in attrs:
            cls._directives = {}

        for attr in attrs.values():
            if isinstance(attr, DirectiveHandler):
                names, handler = attr
                for name in names:
                    cls._directives[name] = handler

    def expand_directive(
        self: Self,
        tokens: Iterator[ParsedToken],
        depth: int,
    ) -> Iterator[ParsedToken]:
        arguments = []

        for token in skipspace(tokens):
            match token.tag:
                case ParsedTokenTag.EOF:
                    errmsg = "unclosed directive"
                    raise PreprocessorError(errmsg, token)

                case ParsedTokenTag.PLAIN if token.data == "{":
                    errmsg = "{ in directive"
                    raise PreprocessorError(errmsg, token)

                case ParsedTokenTag.PLAIN if token.data == "}":
                    end = token
                    break

                case _:
                    token.meta["depth"] = depth
                    arguments.append(token)

        if not arguments:
            errmsg = "empty directive"
            raise PreprocessorError(errmsg, end)

        name = arguments[0]
        handler = self._directives.get(name.data.upper())
        if handler is None:
            errmsg = f"unknown directive: {name.data!r}"
            raise PreprocessorError(errmsg, name)

        try:
            expanded = handler(self, arguments[1:], tokens)
            if expanded is not None:
                yield from expanded

        except DirectiveError as exc:
            token = end if exc.token is None else exc.token
            raise PreprocessorError(exc.message, token) from None


class Preprocessor(Directives):
    def __init__(
        self,
        builder: tree.Builder,
        depth: int = 0,
        defs: MutableMapping[str, Definition] | None = None,
    ) -> None:
        self.builder = builder
        self.depth = depth
        self.defs = ChainMap() if defs is None else ChainMap(defs, {})
        self.compat: str | None = None

    def parse(self, tokens: ParserIO, _depth: int) -> Iterator[ParsedToken]:
        for token in tokens.output:
            match token.meta.get("role"):
                case ParsedTokenRole.OPEN:
                    self.depth += 1
                    self.enter_scope()

                case ParsedTokenRole.CLOSE:
                    self.depth -= 1
                    self.exit_scope()

                case ParsedTokenRole.KEY if self.compat is None:
                    if token.tag == ParsedTokenTag.PLAIN and token.data == "{":
                        expanded = self.expand_directive(
                            tokens.input, self.depth
                        )
                        tokens.input = itertools.chain(expanded, tokens.input)
                        raise RestartParser

                    token = self.evaluate_token(token)

                case ParsedTokenRole.CONDITION:
                    pass

                case ParsedTokenRole.VALUE if self.compat is None:
                    token = self.evaluate_token(token)

                case None if (
                    self.compat is not None
                    and token.tag == ParsedTokenTag.COMMENT
                    and token.data.strip() == self.compat
                ):
                    self.compat = None
                    self.builder.configure(merge=True)
                    # Don't yield this special comment.
                    continue

                case _:
                    pass

            assert "depth" in token.meta
            yield token

    def enter_scope(self) -> None:
        self.defs = self.defs.new_child()
        dbg.log(f"enter scope, depth {len(self.defs.maps) - 1}")

    def exit_scope(self) -> None:
        dbg.log(f"exit scope, depth {len(self.defs.maps) - 1}")
        self.defs = self.defs.parents

    @contextmanager
    def scoped(self) -> Iterator[None]:
        try:
            self.enter_scope()
            yield
        finally:
            self.exit_scope()

    def expand_directive(
        self,
        tokens: Iterator[ParsedToken],
        depth: int,
    ) -> Iterator[ParsedToken]:
        # Builder holds last key and value aside in case a condition follows.
        # The directive might need to access that key, so commit it.
        self.builder.commit()
        return super().expand_directive(tokens, depth)

    def evaluate_token(self, token: ParsedToken) -> ParsedToken:
        assert "depth" in token.meta
        return expr.expand(token, self.evaluate_definition_or_reference)

    def evaluate_definition_or_reference(
        self,
        name: AnyToken,
        arguments: Sequence[AnyToken] | None,
    ) -> ParsedToken:
        if arguments is None:
            return self.evaluate_reference(name)

        return self.evaluate_definition(name, arguments)

    def evaluate_reference(self, path: AnyToken) -> ParsedToken:
        target = self.builder.get().walk(path)

        if isinstance(target, ParsedToken):
            return target

        if target.key is None:
            errmsg = "path refers to name of the root section"
            raise PreprocessorError(errmsg, path)

        return target.key

    def evaluate_definition(
        self,
        name: AnyToken,
        arguments: Sequence[AnyToken],
    ) -> ParsedToken:
        # Recursively expand and parse the definition body, but don't let it
        # affect the state of this preprocessor.
        pp = Preprocessor(self.builder, self.depth, self.defs)
        parser = parser_decorator(pp.parse)(basic_parser)
        # Don't want to parse macros, therefore `depth=1`.
        evaluated = parser(pp.expand_definition(name, arguments), 1)
        result = next(skipspace(evaluated), None)

        if result is None:
            errmsg = "function did not expand to any tokens"
            raise PreprocessorError(errmsg, name)

        assert "depth" in result.meta
        return result

    def expand_definition(
        self,
        name: AnyToken,
        arguments: Sequence[AnyToken],
    ) -> Iterator[ParsedToken]:
        definition = self.defs.get(name.data)

        if definition is None:
            errmsg = f'function "{name.data}" not found'
            raise PreprocessorError(errmsg, name)

        if len(arguments) != definition.arity:
            errmsg = (
                f"function {name.data!r} takes {definition.arity} arguments, "
                f"but {len(arguments) or "none"} were given"
            )
            exc = PreprocessorError(errmsg, name)

            if arguments:
                exc.add_note("Function called with the following arguments:")
                for i, argument in enumerate(arguments):
                    exc.add_note(f"[{i}] {argument.data!r}")

            raise exc

        with self.scoped():
            for param, arg in zip(definition.params, arguments):
                self.defs[param.data] = Definition(
                    params=[],
                    body=[
                        ParsedToken.from_other(
                            arg,
                            tag=ParsedTokenTag.QUOTED,
                            meta={"depth": self.depth},
                        ),
                    ],
                )

            for token in definition.body:
                token = token.clone()
                token.meta["depth"] += self.depth
                yield token

    @Directives.directive("DEFINE", "DEF")
    def do_DEFINE(  # noqa: N802
        self,
        arguments: list[ParsedToken],
        tokens: Iterator[ParsedToken],
    ) -> None:
        if not arguments:
            errmsg = "missing name of defined function"
            raise DirectiveError(errmsg)

        arguments = iter(arguments)
        name = next(arguments)
        params = [self.evaluate_token(token) for token in arguments]

        # Body is stored with depth counting from zero.
        name.meta["depth"] = 0

        body = list(read_balanced(tokens, depth=0))

        self.defs[name.data] = Definition(params, body)

    @Directives.directive("EXPAND")
    def do_EXPAND(  # noqa: N802
        self,
        arguments: list[ParsedToken],
        _tokens: Iterator[ParsedToken],
    ) -> Iterator[ParsedToken]:
        if not arguments:
            errmsg = "missing name of expanded function"
            raise DirectiveError(errmsg)

        arguments = [self.evaluate_token(arg) for arg in arguments]
        return self.expand_definition(arguments[0], arguments[1:])

    @Directives.directive("INHERIT")
    def do_INHERIT(  # noqa: N802
        self,
        arguments: list[ParsedToken],
        _tokens: Iterator[ParsedToken],
    ) -> Iterator[ParsedToken]:
        if not arguments:
            errmsg = "missing name of inherited section"
            raise DirectiveError(errmsg)

        arguments = iter(arguments)
        name = self.evaluate_token(next(arguments))

        parent = self.builder.get().walk(name)
        if not isinstance(parent, KeyValues):
            errmsg = "inherited name does not refer to a section"
            raise DirectiveError(errmsg, name)

        first = next(arguments, None)
        if first is None or (
            first.tag == ParsedTokenTag.PLAIN and first.data.upper() == "EXCEPT"
        ):
            blacklist = {arg.data.lower(): arg for arg in arguments}

            def inherited() -> Iterator[tree.Entry[KeyValues]]:
                for child in parent:
                    key = child.key.data.lower()
                    if key in blacklist:
                        del blacklist[key]
                    else:
                        yield child

                if blacklist:
                    _, missing = next(iter(blacklist.items()))
                    errmsg = f'excluded key "{missing.data}" not found'
                    raise DirectiveError(errmsg, missing)
        else:
            whitelist = itertools.chain([first], arguments)

            def inherited() -> Iterator[tree.Entry[KeyValues]]:
                for key in whitelist:
                    child = parent.get(key)

                    if child is None:
                        errmsg = f'inherited key "{key.data}" not found'
                        raise DirectiveError(errmsg, key)

                    yield child

        for child in inherited():
            yield child.key.clone()

            if child.condition is not None:
                yield child.condition.clone()

            if isinstance(child.value, ParsedToken):
                yield child.value.clone()

            else:
                if child.value.open is not None:
                    yield child.value.open.clone()

                for token in child.value.children.tokens():
                    yield token.clone()

                if child.value.close is not None:
                    yield child.value.close.clone()

    @Directives.directive("FOR")
    def do_FOR(  # noqa: N802
        self,
        arguments: list[ParsedToken],
        tokens: Iterator[ParsedToken],
    ) -> Iterator[ParsedToken]:
        arguments = iter(arguments)

        control_variable_name = next(arguments, None)
        if control_variable_name is None:
            errmsg = "missing control variable name"
            raise DirectiveError(errmsg)

        range_keyword = next(arguments, None)
        if range_keyword is None:
            errmsg = "missing loop range keyword"
            raise DirectiveError(errmsg)

        items: Iterable[ParsedToken]

        match range_keyword.data.upper():
            case "IN":
                items = map(self.evaluate_token, arguments)

            case "BETWEEN":
                start = next(arguments, None)
                if start is None:
                    errmsg = "missing loop start"
                    raise DirectiveError(errmsg)

                start = self.evaluate_token(start)

                try:
                    n_start = Decimal(start.data)
                except decimal.InvalidOperation:
                    errmsg = "invalid loop start"
                    raise DirectiveError(errmsg, start) from None

                end = next(arguments, None)
                if end is None:
                    errmsg = "missing loop end"
                    raise DirectiveError(errmsg)

                end = self.evaluate_token(end)

                try:
                    n_end = Decimal(end.data)
                except decimal.InvalidOperation:
                    errmsg = "invalid loop end"
                    raise DirectiveError(errmsg, end) from None

                step = next(arguments, None)
                if step is None:
                    n_step = Decimal(1 if n_start <= n_end else -1)
                else:
                    step = self.evaluate_token(step)

                    try:
                        n_step = Decimal(step.data)
                    except decimal.InvalidOperation:
                        errmsg = "invalid loop step"
                        raise DirectiveError(errmsg, step) from None

                try:
                    xrange = utils.xrange(n_start, n_end, n_step)
                except ValueError as exc:
                    errmsg = "invalid loop step"
                    raise DirectiveError(errmsg, step) from exc

                items = (
                    ParsedToken(
                        format(n, "f"),
                        tag=ParsedTokenTag.PLAIN,
                        meta={"depth": self.depth},
                    )
                    for n in itertools.chain(xrange, [n_end])
                )

            case _:
                errmsg = "invalid loop range keyword"
                raise DirectiveError(errmsg, range_keyword)

        body = list(read_balanced(tokens, depth=0))
        for token in body:
            token.meta["depth"] += self.depth

        for item in items:
            with self.scoped():
                self.defs[control_variable_name.data] = Definition(
                    params=[],
                    body=[item],
                )

                for token in body:
                    yield token.clone()

    @Directives.directive("DELETE")
    def do_DELETE(  # noqa: N802
        self,
        arguments: list[ParsedToken],
        _tokens: Iterator[ParsedToken],
    ) -> None:
        arguments = map(self.evaluate_token, arguments)

        for key in arguments:
            if not self.builder.get().delete(key):
                errmsg = "key to delete not found"
                raise DirectiveError(errmsg, key)

    @Directives.directive("CLEAR")
    def do_CLEAR(  # noqa: N802
        self,
        arguments: list[ParsedToken],
        _tokens: Iterator[ParsedToken],
    ) -> None:
        arguments = map(self.evaluate_token, arguments)
        whitelist = CaseInsensitiveDict({arg.data: arg for arg in arguments})
        section = self.builder.get()

        section.purge(
            lambda entry: whitelist.pop(entry.key.data, None) is not None,
        )

        if whitelist:
            _, token = whitelist.popitem()
            errmsg = f'whitelisted key "{token.data}" not found'
            raise DirectiveError(errmsg, token)

    @Directives.directive("MOVE")
    def do_MOVE(  # noqa: N802
        self,
        arguments: list[ParsedToken],
        _tokens: Iterator[ParsedToken],
    ) -> None:
        arguments = map(self.evaluate_token, arguments)
        section = self.builder.get()

        for key in arguments:
            entry = section.get(key)

            if entry is None:
                errmsg = f'key "{key.data}" not found'
                raise DirectiveError(errmsg, key)

            section.delete(entry.key, entry.condition)
            section.append(entry)

    @Directives.directive("PRAGMA")
    def do_PRAGMA(  # noqa: N802
        self,
        arguments: list[ParsedToken],
        _tokens: Iterator[ParsedToken],
    ) -> None:
        if not arguments:
            errmsg = "missing pragma name"
            raise DirectiveError(errmsg)

        name = arguments[0]

        match name.data.upper():
            case "COMPAT":
                if len(arguments) != 2:
                    errmsg = "PRAGMA COMPAT expects one argument"
                    raise DirectiveError(errmsg)

                self.compat = arguments[1].data
                self.builder.configure(merge=False)

            case _:
                errmsg = f'unknown pragma "{name.data}"'
                raise DirectiveError(errmsg, name)


def preprocess(parser: ParserFn) -> ParserFn:
    builder = tree.Builder()
    preprocessor = Preprocessor(builder)

    @parser_decorator
    def expand(
        tokens: ParserIO,
        depth: int,
    ) -> Iterator[ParsedToken]:
        return preprocessor.parse(tokens, depth)

    @parser_decorator
    def build(
        tokens: ParserIO,
        _depth: int,
    ) -> Iterator[ParsedToken]:
        for token in tokens.output:
            builder.token(token)

        for token in builder.get().tokens():
            yield token

    return build(expand(parser))
