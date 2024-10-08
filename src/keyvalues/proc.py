from __future__ import annotations

import collections
import dataclasses
import decimal
import functools
import io
import itertools
import operator
import re
import typing
from collections.abc import (
    Callable,
    Iterable,
    Iterator,
    Sequence,
)
from dataclasses import dataclass
from decimal import Decimal
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Self,
    TypeGuard,
    assert_never,
    overload,
)

from . import utils
from .fmt import writer
from .parse import (
    Token,
    TokenError,
    TokenRole,
    TokenTag,
    is_condition,
    parser_decorator,
    skipspace,
    yieldspace,
)

if TYPE_CHECKING:
    from types import EllipsisType

    from .parse import ParserFn


class KeyValuesPreprocessorError(TokenError):
    pass


def read_balanced(tokens: Iterable[Token], depth: int = 0) -> Iterable[Token]:
    # Yield leading whitespace and comments.
    nonspace = yield from yieldspace(tokens, depth)

    # Yield tokens until we reach zero depth.
    for token in itertools.chain([nonspace], tokens):
        token.depth = depth

        match token.tag:
            case TokenTag.EOF:
                errmsg = "missing }"
                raise KeyValuesPreprocessorError(errmsg, token)

            case TokenTag.PLAIN:
                match token.data:
                    case "{":
                        depth += 1
                    case "}":
                        depth -= 1
                        # Braces are not considered to be inside their section.
                        token.depth = depth

        yield token

        if depth == 0:
            break


def read_directive(tokens: Iterable[Token]) -> Iterator[Token | list[Token]]:
    # No space in topmost section.
    for token in skipspace(tokens):
        match token.tag:
            case TokenTag.EOF:
                errmsg = "unclosed directive"
                raise KeyValuesPreprocessorError(errmsg, token)

            case TokenTag.PLAIN:
                match token.data:
                    case "{":
                        # Capture space an nested sections.
                        yield [token, *read_balanced(tokens, depth=1)]
                        continue

                    case "}":
                        yield token
                        break

        yield token


def yield_directive(*tokens: Token) -> Iterator[Token]:
    yield Token("{", tag=TokenTag.PLAIN)
    yield from tokens
    yield Token("}", tag=TokenTag.PLAIN)


def is_range_finite(start: Any, end: Any, step: Any) -> Any:
    return step > 0 if start <= end else step < 0


def xrange(start: Any, end: Any, step: Any) -> Iterator[Any]:
    if step == 0:
        errmsg = "step must be nonzero"
        raise ValueError(errmsg)

    if step > 0:
        while start < end:
            yield start
            start += step
    else:
        while start > end:
            yield start
            start += step


@dataclass
class Definition:
    params: list[Token]
    content: list[Token]
    arity: int = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.arity = len(self.params)


@dataclass(slots=True)
class Directive:
    begin: Token
    content: list[Token | list[Token]]
    end: Token

    @overload
    def __getitem__(self, index: int) -> Token | list[Token]: ...

    @overload
    def __getitem__(self, index: slice) -> list[Token | list[Token]]: ...

    def __getitem__(self, index: int | slice) -> Any:
        return self.content[index]

    def __iter__(self) -> Iterator[Token | list[Token]]:
        return iter(self.content)

    def __len__(self) -> int:
        return len(self.content)

    @classmethod
    def read(cls, first: Token, tokens: Iterable[Token]) -> Self:
        content: list[Token | list[Token]] = []

        for token in skipspace(tokens):
            match token.tag:
                case TokenTag.EOF:
                    errmsg = "unclosed directive"
                    raise KeyValuesPreprocessorError(errmsg, token)

                case TokenTag.PLAIN:
                    match token.data:
                        case "{":
                            # Capture space an nested sections.
                            content.append(
                                [token, *read_balanced(tokens, depth=1)]
                            )
                            continue

                        case "}":
                            end = token
                            break

            content.append(token)

        if not content:
            errmsg = "empty directive"
            raise KeyValuesPreprocessorError(errmsg, end)

        return cls(first, content, end)


if TYPE_CHECKING:
    LookupMap = dict[str, Token | LookupMap]  # noqa: F821


def follow_path(lookup: Iterable[LookupMap], path: Token) -> LookupMap | Token:
    stack: list[LookupMap | Token] = list(lookup)

    for key in path.split("/"):
        dots = key.data.count(".")
        if dots == len(key.data):
            for _ in range(dots - 1):
                stack.pop()
        else:
            parent = stack[-1]

            if isinstance(parent, Token):
                errmsg = "invalid path"
                raise KeyValuesPreprocessorError(errmsg, path)

            child = parent.get(key.data.lower())

            if child is None:
                errmsg = f'key "{key.data}" not found'
                raise KeyValuesPreprocessorError(errmsg, key)

            stack.append(child)

    return stack[-1]


class TokenFlags(utils.AutoFlagEnum):
    EXPAND = ()
    OVERRIDE = ()

    def parse(self, *tokens: Token) -> TokenFlags:
        flags = self

        for token in tokens:
            flag = token.data.upper()

            negate = flag.startswith("NO")
            if negate:
                flag = flag[2:]

            try:
                iflag = TokenFlags[flag]
            except KeyError:
                errmsg = f'invalid flag "{flag}"'
                raise KeyValuesPreprocessorError(errmsg, token) from None
            else:
                if negate:
                    flags &= ~iflag
                else:
                    flags |= iflag

        return flags


class Action(utils.AutoIntEnum):
    CLEAR = ()
    DELETE = ()
    MOVE = ()

    def to_comments(self, *payload: Token) -> Iterable[Token]:
        yield Token(
            f"<{self.name}>",
            tag=TokenTag.COMMENT,
            flags=TokenFlags.EXPAND,
        )

        for token in payload:
            token.tag = TokenTag.COMMENT
            yield token

        yield Token(
            f"</{self.name}>",
            tag=TokenTag.COMMENT,
            flags=TokenFlags.EXPAND,
        )

    @staticmethod
    def from_comments(
        first: Token,
        tokens: Iterable[Token],
    ) -> tuple[Action, list[Token]]:
        name = first.data[1:-1]
        action = Action[name]

        payload = []
        last = None

        for token in tokens:
            assert token.tag is TokenTag.COMMENT

            if TokenFlags.EXPAND & token.flags:
                last = token
                break

            # The original tag is lost, QUOTED is a safe choice.
            token.tag = TokenTag.QUOTED
            payload.append(token)

        assert last is not None
        assert last.data == f"</{name}>"

        return action, payload


# "NO" prefix is used to negate flags.
assert all(not name.startswith("NO") for name in TokenFlags.__members__)


def is_flat(items: list[T | list[T]]) -> TypeGuard[list[T]]:
    return not any(isinstance(item, list) for item in items)


def first_nested(items: list[T | list[T]]) -> list[T]:
    for item in items:
        if isinstance(item, list):
            return item

    errmsg = "list is flat"
    raise ValueError(errmsg)


class expand:  # noqa: N801
    def __init__(self, parser: ParserFn):
        functools.update_wrapper(self, parser)

        self.parser = parser
        self.defs = collections.ChainMap[str, Definition]()
        self.lookup: list[LookupMap] = [{}]
        self.default_flags = TokenFlags(0)
        self.compat: str | None = None

    def __call__(self, tokens: Iterable[Token], depth: int) -> Iterator[Token]:
        last_key: Token | None = None

        repeat = True
        while repeat:
            repeat = False

            for token in self.parser(tokens, depth):
                depth = token.depth

                match token.role:
                    case TokenRole.OPEN:
                        self.enter_scope()

                        parent = self.lookup[-1]

                        assert last_key is not None
                        child = parent.get(last_key.data.lower())

                        if not isinstance(child, dict):
                            # "." stores the key of this section.
                            child = {".": last_key}
                            parent[last_key.data.lower()] = child

                        self.lookup.append(child)

                    case TokenRole.CLOSE:
                        self.exit_scope()

                        self.lookup.pop()
                        assert self.lookup, "unmatched open/close"

                    case TokenRole.KEY:
                        if self.compat is not None:
                            token.flags = self.default_flags
                            last_key = token
                            expect_modifier = False

                        elif token.tag is TokenTag.PLAIN and token.data == "{":
                            # Directive is read from the unparsed token stream.
                            directive = Directive.read(token, tokens)
                            expanded = self.expand_directive(directive, tokens)

                            # Prepend expanded directive and restart the parser.
                            tokens = itertools.chain(expanded, tokens)
                            repeat = True
                            break

                        else:
                            if TokenFlags.EXPAND & self.default_flags:
                                token = expand_expressions(
                                    token, self.evaluate_definition
                                )

                            token.flags = self.default_flags
                            last_key = token
                            expect_modifier = True

                    case TokenRole.VALUE:
                        expect_modifier = False

                        assert last_key is not None

                        if TokenFlags.EXPAND & last_key.flags:
                            token = expand_expressions(
                                token, self.evaluate_definition
                            )

                        self.lookup[-1][last_key.data.lower()] = token

                    case TokenRole.CONDITION if expect_modifier:
                        expect_modifier = False

                        if re.fullmatch(r"\$\[.*\]", token.data):
                            subtokens = token[2:-1].split(r",")

                            assert last_key is not None
                            assert isinstance(last_key.flags, TokenFlags)
                            last_key.flags = last_key.flags.parse(*subtokens)

                            # Don't yield this token.
                            continue

                    case None if (
                        token.tag is TokenTag.COMMENT
                        and self.compat is not None
                        and token.data.strip() == self.compat
                    ):
                        self.compat = None
                        continue

                yield token

    def expand_directive(
        self,
        directive: Directive,
        tokens: Iterable[Token],
    ) -> Iterator[Token]:
        directive.content = [
            expand_expressions(outer, self.evaluate_definition)
            if isinstance(outer, Token)
            else [
                expand_expressions(inner, self.evaluate_definition)
                for inner in outer
            ]
            for outer in directive.content
        ]

        first = directive[0]
        if isinstance(first, list):
            errmsg = "invalid directive name"
            raise KeyValuesPreprocessorError(errmsg, first[0])

        match first.data.upper():
            case "COMMENT":
                pass

            case "BEGIN":
                self.enter_scope()

            case "END":
                self.exit_scope()

            case "DEFINE":
                self.store_definition(directive, tokens)

            case "EXPAND":
                if len(directive) < 2:
                    errmsg = "missing function name"
                    raise KeyValuesPreprocessorError(errmsg, directive.end)

                name = directive[1]
                if isinstance(name, list):
                    errmsg = "invalid function name"
                    raise KeyValuesPreprocessorError(errmsg, name[0])

                arguments = directive[2:]
                yield from self.expand_definition(name, arguments, 1)

            case "INHERIT":
                if len(directive) < 2:
                    errmsg = "missing inherited path"
                    raise KeyValuesPreprocessorError(errmsg, directive.end)

                path = directive[1]
                if isinstance(path, list):
                    errmsg = "invalid INHERIT source path"
                    raise KeyValuesPreprocessorError(errmsg, path[0])

                keys = directive[2:]
                if not is_flat(keys):
                    section = first_nested(keys)
                    errmsg = "invalid INHERIT key"
                    raise KeyValuesPreprocessorError(errmsg, section[0])

                yield from self.expand_inherit(path, keys)

            case "FOR":
                yield from self.expand_loop(directive, tokens)

            case "CLEAR":
                whitelist = directive[1:]

                if not is_flat(whitelist):
                    section = first_nested(whitelist)
                    errmsg = "invalid CLEAR key"
                    raise KeyValuesPreprocessorError(errmsg, section[0])

                yield from Action.CLEAR.to_comments(*whitelist)

            case "MOVE":
                keys = directive[1:]
                if not is_flat(keys):
                    section = first_nested(keys)
                    errmsg = "invalid MOVE key"
                    raise KeyValuesPreprocessorError(errmsg, section[0])

                yield from Action.MOVE.to_comments(*keys)

            case "DELETE":
                if len(directive) < 2:
                    errmsg = "missing keys to delete"
                    raise KeyValuesPreprocessorError(errmsg, directive.end)

                blacklist = directive[1:]

                if not is_flat(blacklist):
                    section = first_nested(blacklist)
                    errmsg = "invalid DELETE key"
                    raise KeyValuesPreprocessorError(errmsg, section[0])

                yield from Action.DELETE.to_comments(*blacklist)

            case "PRAGMA":
                pragma = directive[1:]

                if not pragma:
                    errmsg = "empty pragma"
                    raise KeyValuesPreprocessorError(errmsg, first)

                if not is_flat(pragma):
                    section = first_nested(pragma)
                    errmsg = "invalid pragma directive"
                    raise KeyValuesPreprocessorError(errmsg, section[0])

                self.apply_pragma(pragma)

            case _:
                errmsg = f'invalid directive "{first.data}"'
                raise KeyValuesPreprocessorError(errmsg, first)

    def apply_pragma(self, pragma: list[Token]) -> None:
        name = pragma[0]

        match name.data.upper():
            case "FLAGS":
                self.default_flags = self.default_flags.parse(*pragma[1:])

            case "COMPAT":
                if len(pragma) != 2:
                    errmsg = "pragma COMPAT expects one argument"
                    raise KeyValuesPreprocessorError(errmsg, name)

                self.compat = pragma[1].data

            case _:
                errmsg = f'unknown pragma "{name.data}"'
                raise KeyValuesPreprocessorError(errmsg, name)

    def enter_scope(self) -> None:
        self.defs = self.defs.new_child()

    def exit_scope(self) -> None:
        self.defs = self.defs.parents

    def store_definition(
        self,
        directive: Directive,
        tokens: Iterable[Token],
    ) -> None:
        if len(directive) < 2:
            errmsg = "missing function name"
            raise KeyValuesPreprocessorError(errmsg, directive.end)

        name = directive[1]
        if isinstance(name, list):
            errmsg = "invalid function name"
            raise KeyValuesPreprocessorError(errmsg, name[0])

        params = directive[2:]
        if not is_flat(params):
            section = first_nested(params)
            errmsg = "invalid function parameter"
            raise KeyValuesPreprocessorError(errmsg, section[0])

        content = [name.clone(depth=0)]
        # Read next condition (optional) and value (or section).
        content.extend(read_balanced(tokens))
        if is_condition(content[-1]):
            content.extend(read_balanced(tokens))

        self.defs[name.data] = Definition(params, content)

    def expand_definition(
        self,
        name: Token,
        arguments: Sequence[Token | Sequence[Token]],
        min_depth: int = 0,
    ) -> Iterator[Token]:
        definition = self.defs.get(name.data)
        if definition is None:
            errmsg = f'function "{name.data}" not found'
            raise KeyValuesPreprocessorError(errmsg, name)

        if len(arguments) != definition.arity:
            errmsg = (
                f'function "{name.data}" takes {definition.arity}'
                f" arguments, but {len(arguments)} were given"
            )
            e = KeyValuesPreprocessorError(errmsg, name)

            if arguments:
                e.add_note("Function was called with the following arguments:")
                for i, argument in enumerate(arguments, 1):
                    note = io.StringIO()
                    note.write(f"[{i}] ")
                    if isinstance(argument, Token):
                        note.write(str(argument))
                    else:
                        writer(argument, note)
                    e.add_note(note.getvalue())

            raise e

        # Create a new scope for the expanded function.
        yield from yield_directive(Token("BEGIN", tag=TokenTag.PLAIN))

        # Define a nullary local function for each passed argument.
        for param, arg in zip(definition.params, arguments):
            yield from yield_directive(
                Token("DEFINE", tag=TokenTag.PLAIN),
                Token(" ", tag=TokenTag.SPACE),
                param,
            )
            if isinstance(arg, Token):
                yield arg
            else:
                yield from arg

        # Yield only tokens inside a section.
        for token in definition.content:
            if token.depth >= min_depth:
                yield token.clone()

        # Close the scope created for the expanded function.
        yield from yield_directive(Token("END", tag=TokenTag.PLAIN))

    def evaluate_definition(
        self,
        name: Token,
        arguments: Sequence[Token] | None,
    ) -> Token:
        if arguments is not None:
            tokens = self.expand_definition(name, arguments, 0)

            # Function calls in expressions should not affect later lookups.
            lookup = self.lookup.copy()

            for token in self(tokens, 0):
                if token.role is TokenRole.VALUE:
                    self.lookup = lookup
                    return token

            errmsg = "function did not produce any value"
            raise KeyValuesPreprocessorError(errmsg, name)

        # Wihout arguments, evaluate name as relative path.

        value = follow_path(self.lookup, name)
        if isinstance(value, Token):
            return value

        # Trailing "." refers to the section name.
        value = value.get(".")
        if value is None:
            errmsg = "path refers to root section"
            raise KeyValuesPreprocessorError(errmsg, name)

        assert isinstance(value, Token)
        return value

    def expand_inherit(
        self,
        path: Token,
        keys: list[Token],
    ) -> Iterator[Token]:
        source = follow_path(self.lookup, path)
        if isinstance(source, Token):
            errmsg = "INHERIT source path leads to value"
            raise KeyValuesPreprocessorError(errmsg, path)

        if keys:
            for key in keys:
                value = source.get(key.data.lower())

                if value is None:
                    errmsg = f'INHERIT key "{key.data}" not found in source'
                    raise KeyValuesPreprocessorError(errmsg, path)

                if not isinstance(value, Token):
                    errmsg = f'INHERIT key "{key.data}" refers to a section'
                    raise KeyValuesPreprocessorError(errmsg, path)

                yield key
                yield value

        else:
            for key_data, value in source.items():
                if key_data != "." and isinstance(value, Token):
                    yield Token(key_data, tag=TokenTag.QUOTED)
                    yield value

    def expand_loop(
        self,
        directive: Directive,
        tokens: Iterable[Token],
    ) -> Iterator[Token]:
        length = len(directive)

        if length < 3:
            errmsg = "invalid FOR directive"
            raise KeyValuesPreprocessorError(errmsg, directive.end)

        name = directive[1]
        if isinstance(name, list):
            errmsg = "invalid loop control variable name"
            raise KeyValuesPreprocessorError(errmsg, name[0])

        loop_kind = directive[2]
        if isinstance(loop_kind, list):
            errmsg = "invalid FOR directive"
            raise KeyValuesPreprocessorError(errmsg, loop_kind[0])

        items: Iterable[Token | list[Token]]

        match loop_kind.data.upper():
            case "IN":
                items = directive[3:]

            case "BETWEEN":
                if length < 4:
                    errmsg = "missing loop start"
                    raise KeyValuesPreprocessorError(errmsg, directive.end)

                start = directive[3]
                if isinstance(start, list):
                    errmsg = "invalid loop start"
                    raise KeyValuesPreprocessorError(errmsg, start[0])

                try:
                    n_start = Decimal(start.data)
                except decimal.InvalidOperation:
                    errmsg = "invalid loop start"
                    raise KeyValuesPreprocessorError(errmsg, start) from None

                if length < 5:
                    errmsg = "missing loop end"
                    raise KeyValuesPreprocessorError(errmsg, directive.end)

                end = directive[4]
                if isinstance(end, list):
                    errmsg = "invalid loop end"
                    raise KeyValuesPreprocessorError(errmsg, end[0])

                try:
                    n_end = Decimal(end.data)
                except decimal.InvalidOperation:
                    errmsg = "invalid loop end"
                    raise KeyValuesPreprocessorError(errmsg, end) from None

                if length > 5:
                    step = directive[5]

                    if isinstance(step, list):
                        errmsg = "invalid loop step"
                        raise KeyValuesPreprocessorError(errmsg, step[0])

                    try:
                        n_step = Decimal(step.data)
                    except decimal.InvalidOperation:
                        errmsg = "invalid loop step"
                        raise KeyValuesPreprocessorError(errmsg, step) from None

                    if not is_range_finite(n_start, n_end, n_step):
                        errmsg = "loop step results in an infinite loop"
                        raise KeyValuesPreprocessorError(errmsg, step)

                else:
                    n_step = Decimal(1 if n_start <= n_end else -1)

                items = (
                    Token(format(n, "f"), tag=TokenTag.PLAIN)
                    for n in xrange(n_start, n_end, n_step)
                )

            case _:
                errmsg = "invalid FOR directive"
                raise KeyValuesPreprocessorError(errmsg, loop_kind)

        # Get loop body without surrounding braces.
        body = [token for token in read_balanced(tokens) if token.depth > 0]

        yield from yield_directive(Token("BEGIN", tag=TokenTag.PLAIN))

        for item in items:
            yield from yield_directive(
                Token("DEFINE", tag=TokenTag.PLAIN),
                Token(" ", tag=TokenTag.SPACE),
                name,
            )

            if isinstance(item, list):
                yield from item
            else:
                yield item

            for token in body:
                yield token.clone()

        yield from yield_directive(Token("END", tag=TokenTag.PLAIN))


class ExpressionTag(utils.RegexEnum):
    # fmt: off
    AND    = r"&&"
    OR     = r"\|\|"

    BNOT   = r"~"
    BAND   = r"&"
    BOR    = r"\|"

    LSHIFT = r"<<"
    RSHIFT = r">>"

    EQ     = r"=="
    NE     = r"!="
    NOT    = r"!"

    GE     = r">="
    GT     = r">"
    LE     = r"<="
    LT     = r"<"

    PLUS   = r"\+"
    MINUS  = r"-"
    MULT   = r"\*"
    DIVIDE = r"/"
    MODULO = r"%"
    POWER  = r"\^"

    EXPAND = r"\$"

    OPEN   = r"\("
    CLOSE  = r"\)"

    NUMBER = r"[+-]?(?:\.\d+|\d+(?:\.\d*)?)"
    PATH   = r"\.[/.\w]*"
    NAME   = r"[^\W\d][\w.]*"

    SPACE  = r"\s+"
    ERROR  = r".+?"  # only single character, unless matching whole string
    EOF    = r"$"
    # fmt: on


class KeyValuesExpressionError(TokenError):
    pass


def tokenize_expression(expression: Token) -> Iterator[Token]:
    for match in ExpressionTag.finditer(expression.data):
        assert match.lastgroup is not None
        tag = ExpressionTag[match.lastgroup]

        if tag is ExpressionTag.SPACE:
            continue

        token = expression[match.start() : match.end()]
        token.tag = tag

        if tag is ExpressionTag.ERROR:
            errmsg = f'unexpected character "{token.data}" in expression'
            raise KeyValuesExpressionError(errmsg, token)

        yield token


def tokenize_arguments(expression: Token) -> Iterator[Token]:
    tokens = tokenize_expression(expression)

    for token in tokens:
        match token.tag:
            case ExpressionTag.EOF:
                return

            case ExpressionTag.PLUS | ExpressionTag.MINUS:
                # Join sign and number into a single token.
                number = next(tokens)

                if number.tag is not ExpressionTag.NUMBER:
                    break

                number.data = token.data + number.data
                yield number

            case ExpressionTag.NUMBER | ExpressionTag.NAME:
                yield token

            case _:
                break

    errmsg = f'invalid subtoken "{token.data}" in expanded token'
    raise KeyValuesExpressionError(errmsg, token)


def to_decimal(token: Token) -> Decimal:
    if token.tag is ExpressionTag.NUMBER:
        return Decimal(token.data)

    errmsg = f'"{token}" cannot be converted to a number'
    raise KeyValuesExpressionError(errmsg, token)


class Operator:
    def __init__(
        self,
        op: Callable[..., Any],
        bp: int,
        *,
        infixl: bool = True,
        cast: Callable[[Decimal], Any] | None = None,
    ):
        self.op = op
        self.lbp = bp * 2 + (1 - infixl)
        self.rbp = bp * 2
        self.cast = cast if cast is not None else lambda x: x

    def invoke(self, token: Token, *args: Token) -> Token:
        value = self.op(*map(self.cast, map(to_decimal, args)))
        if not isinstance(value, Decimal):
            value = Decimal(value)

        token = token.clone(data=format(value, "f"))
        token.tag = ExpressionTag.NUMBER
        return token


PREFIX_OPERATORS = {
    ExpressionTag.PLUS: Operator(operator.pos, 9),
    ExpressionTag.MINUS: Operator(operator.neg, 9),
    ExpressionTag.NOT: Operator(operator.not_, 9),
    ExpressionTag.BNOT: Operator(operator.inv, 9, cast=int),
}

INFIX_OPERATORS = {
    ExpressionTag.AND: Operator(lambda x, y: x and y, 0),
    ExpressionTag.OR: Operator(lambda x, y: x or y, 1),
    ExpressionTag.EQ: Operator(operator.eq, 2),
    ExpressionTag.NE: Operator(operator.ne, 2),
    ExpressionTag.LE: Operator(operator.le, 3),
    ExpressionTag.LT: Operator(operator.lt, 3),
    ExpressionTag.GE: Operator(operator.ge, 3),
    ExpressionTag.GT: Operator(operator.gt, 3),
    ExpressionTag.BAND: Operator(operator.and_, 4, cast=int),
    ExpressionTag.BOR: Operator(operator.or_, 5, cast=int),
    ExpressionTag.LSHIFT: Operator(operator.lshift, 6, cast=int),
    ExpressionTag.RSHIFT: Operator(operator.rshift, 6, cast=int),
    ExpressionTag.PLUS: Operator(operator.add, 7),
    ExpressionTag.MINUS: Operator(operator.sub, 7),
    ExpressionTag.MULT: Operator(operator.mul, 8),
    ExpressionTag.DIVIDE: Operator(operator.truediv, 8),
    ExpressionTag.MODULO: Operator(operator.mod, 8),
    ExpressionTag.POWER: Operator(operator.pow, 9, infixl=False),
}


T = typing.TypeVar("T")


class Peekable(Generic[T], Iterator[T]):
    def __init__(self, iterable: Iterable[T]):
        self._it = iter(iterable)
        self._next = collections.deque[T]()

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> T:
        return self._next.popleft() if self._next else next(self._it)

    def __getitem__(self, index: int) -> T:
        if index < 0:
            errmsg = "negative index"
            raise ValueError(errmsg)

        self._load(index + 1)
        return self._next[index]

    def get(self, index: int, default: T | None = None) -> T | None:
        try:
            return self[index]
        except IndexError:
            return default

    def skip(self, count: int) -> None:
        next(itertools.islice(self, count, count), None)

    def _load(self, count: int) -> None:
        loaded = len(self._next)
        if count > loaded:
            self._next.extend(itertools.islice(self._it, count - loaded))


def pratt(
    tokens: Peekable[Token],
    expander: Callable[[Token, Sequence[Token] | None], Token],
    min_bp: int = 0,
    *,
    take_one: bool = False,
    stop_on_expand: bool = False,
) -> Token:
    recurse = functools.partial(pratt, tokens, expander)

    token = next(tokens)

    match token.tag:
        case ExpressionTag.NUMBER:
            lhs = token

        case ExpressionTag.OPEN:
            lhs = recurse(0)
            if next(tokens).tag is not ExpressionTag.CLOSE:
                errmsg = "unclosed ("
                raise KeyValuesExpressionError(errmsg, token)

        # Parse prefix operators (including function calls).

        case ExpressionTag.NAME:
            arguments = []
            while True:
                match tokens[0].tag:
                    case ExpressionTag.OPEN:
                        arguments.append(recurse(0, take_one=True))

                    case (
                        ExpressionTag.NUMBER
                        | ExpressionTag.NAME
                        | ExpressionTag.PATH
                    ):
                        arguments.append(next(tokens))

                    case ExpressionTag.EXPAND if not stop_on_expand:
                        next(tokens)
                        value = recurse(0, take_one=True, stop_on_expand=True)
                        arguments.extend(tokenize_arguments(value))

                    case _:
                        break

            lhs = expander(token, arguments)

        case ExpressionTag.PATH:
            lhs = expander(token, None)

        case _:
            op = PREFIX_OPERATORS.get(ExpressionTag(token.tag))
            if op is None:
                errmsg = f'unexpected token "{token.data}" in expression'
                raise KeyValuesExpressionError(errmsg, token)

            rhs = recurse(op.rbp)
            lhs = op.invoke(token, rhs)

    if take_one:
        return lhs

    while True:
        token = tokens[0]

        match token.tag:
            case ExpressionTag.EOF:
                break

            case ExpressionTag.CLOSE:
                break

            # Parse infix operators.

            case _:
                op = INFIX_OPERATORS.get(ExpressionTag(token.tag))
                if op is None:
                    errmsg = f'expected operator, got "{token}"'
                    raise KeyValuesExpressionError(errmsg, token)

                if op.lbp <= min_bp:
                    break

                next(tokens)

                rhs = recurse(op.rbp)
                lhs = op.invoke(token, lhs, rhs)

    return lhs


def evaluate_expression(
    expression: Token,
    expander: Callable[[Token, Sequence[Token] | None], Token],
) -> Token:
    def expr_expander(name: Token, arguments: Sequence[Token] | None) -> Token:
        token = expander(name, arguments)

        # Set correct tag.
        match = ExpressionTag.fullmatch(token.data)
        assert match is not None, "at least one group should always match"
        assert match.lastgroup is not None
        token.tag = ExpressionTag[match.lastgroup]

        return token

    try:
        result = pratt(Peekable(tokenize_expression(expression)), expr_expander)
    except TokenError as e:
        errmsg = "error while evaluating expression"
        raise KeyValuesPreprocessorError(errmsg, expression) from e

    return result.clone(tag=TokenTag.QUOTED)


def expand_expressions(
    token: Token,
    expander: Callable[[Token, Sequence[Token] | None], Token],
) -> Token:
    parts = []

    cursor = 0
    start = 0
    depth = 0

    pattern = re.compile(r"[()]|\$(?:\$|(?=\(|\w))")
    while match := pattern.search(token.data, cursor):
        cursor = match.end()

        match match[0]:
            case "$$" if depth == 0:
                parts.append(token.data[start : cursor - 1])
                start = cursor
                continue

            case "$" if depth == 0:
                parts.append(token.data[start : cursor - 1])
                start = cursor

                if token.data[cursor] == "(":
                    depth = 1
                    cursor += 1
                elif nonword := re.search(r"\W", token.data, cursor):
                    cursor = nonword.start()
                else:
                    cursor = len(token.data)

            case "(" if depth > 0:
                depth += 1

            case ")" if depth > 0:
                depth -= 1

            case _:
                continue

        if depth == 0:
            expression = token[start:cursor]
            parts.append(evaluate_expression(expression, expander).data)
            start = cursor

    parts.append(token.data[start:])
    return token.clone(data="".join(parts), tag=TokenTag.QUOTED)


@dataclass(slots=True)
class KeyConditionValue(Generic[T]):
    key: Token
    condition: Token | None
    value: Token | T

    def unpack(self) -> tuple[Token, Token | None, Token | T]:
        return self.key, self.condition, self.value


class MergedKeyValues:
    def __init__(self) -> None:
        self.open: Token | None = None
        self.items: list[KeyConditionValue[MergedKeyValues] | None] = []
        self.close: Token | None = None

        # Indices into `self.items`.
        self.by_key: dict[str, int] = {}
        self.by_key_and_condition: dict[tuple[str, str | None], int] = {}

    def get(
        self,
        key: str | Token,
        condition: str | Token | None | EllipsisType = Ellipsis,
    ) -> tuple[int, KeyConditionValue[MergedKeyValues] | None]:
        if isinstance(key, Token):
            key = key.data
        if isinstance(condition, Token):
            condition = condition.data

        key = key.lower()
        if isinstance(condition, str):
            condition = condition.lower()

        if condition is Ellipsis:
            index = self.by_key.get(key)
        else:
            index = self.by_key_and_condition.get((key, condition))

        if index is None:
            return -1, None

        item = self.items[index]
        return index, item

    def pop(self, index: int) -> KeyConditionValue[MergedKeyValues] | None:
        item = self.items[index]

        if item is None:
            return None

        self.items[index] = None

        key, condition, _ = item.unpack()

        key_str = key.data.lower()
        condition_str = None if condition is None else condition.data.lower()

        del self.by_key[key_str]
        del self.by_key_and_condition[key_str, condition_str]

        return item

    def append(
        self,
        item: KeyConditionValue[MergedKeyValues],
    ) -> int:
        key, condition, value = item.unpack()

        key_str = key.data.lower()
        condition_str = None if condition is None else condition.data.lower()

        if TokenFlags.OVERRIDE & key.flags:
            index, other = self.get(key_str, condition_str)
            if other is not None:
                if isinstance(other.value, MergedKeyValues) and isinstance(
                    value, MergedKeyValues
                ):
                    other.value.merge(value)
                else:
                    other.value = value

                return index

        # Clear the OVERRIDE flag to skip the above branch when moving items.
        key.flags &= ~TokenFlags.OVERRIDE

        index = len(self.items)
        self.by_key[key_str] = index
        self.by_key_and_condition[key_str, condition_str] = index

        self.items.append(item)
        return index

    def merge(self, other: MergedKeyValues) -> None:
        for item in other.items:
            if item is not None:
                self.append(item)

    def parse(self, tokens: Iterable[Token], depth: int = 0) -> None:
        """Load contents from parsed tokens.

        Assumes that `tokens` are well-formed.
        """
        filtered = self.filter_actions(tokens)
        key = next(filtered)

        while key.role is TokenRole.KEY:
            value = next(filtered)

            if value.role is TokenRole.CONDITION:
                condition = value
                value = next(filtered)
            else:
                condition = None

            match value.role:
                case TokenRole.OPEN:
                    if TokenFlags.OVERRIDE & key.flags:
                        _, item = self.get(key, condition)
                        section = None if item is None else item.value
                    else:
                        section = None

                    if isinstance(section, MergedKeyValues):
                        # Parse into an existing section.
                        section.parse(tokens, depth + 1)
                    else:
                        section = MergedKeyValues()
                        section.open = value
                        section.parse(tokens, depth + 1)
                        self.append(KeyConditionValue(key, condition, section))

                    # The recursive call might have read the EOF token.
                    nextkey = next(filtered, None)
                    if nextkey is None:
                        break

                case TokenRole.VALUE:
                    # Peeking ahead might read action comments which require
                    # the item to already be appended (they might reference it).
                    index = self.append(
                        KeyConditionValue(key, condition, value)
                    )

                    nextkey = next(filtered)
                    if nextkey.role is TokenRole.CONDITION:
                        last_item = self.items[index]
                        assert last_item is not None, "just appended this"
                        last_item.condition = nextkey
                        nextkey = next(filtered)

                case _:
                    errmsg = "unexpected token role"
                    raise AssertionError(errmsg)

            key = nextkey

        if key.role is TokenRole.CLOSE:
            self.close = key

    def filter_actions(self, tokens: Iterable[Token]) -> Iterator[Token]:
        """Filter out semantically irrelevant tokens and apply action comments.

        Actions are applied on `self`. Filtered tokens must not be passed to
        nested instances!
        """
        for token in tokens:
            if token.role is not None:
                yield token

            match token.tag:
                case TokenTag.EOF:
                    yield token

                case TokenTag.COMMENT if TokenFlags.EXPAND & token.flags:
                    self.apply_action(token, tokens)

                case _:
                    pass

    def apply_action(self, token: Token, tokens: Iterable[Token]) -> None:
        action, arguments = Action.from_comments(token, tokens)

        match action:
            case Action.DELETE:
                for key in arguments:
                    index, item = self.get(key)

                    if item is None:
                        errmsg = f'{action.name}: key "{key.data}" not found'
                        raise KeyValuesPreprocessorError(errmsg, key)

                    self.pop(index)

            case Action.CLEAR:
                indices_to_delete = [True] * len(self.items)

                # Undelete selected items.
                for key in arguments:
                    index, item = self.get(key)

                    if item is None:
                        errmsg = f'{action.name}: key "{key.data}" not found'
                        raise KeyValuesPreprocessorError(errmsg, key)

                    indices_to_delete[index] = False

                for index, should_delete in enumerate(indices_to_delete):
                    if should_delete:
                        self.pop(index)

            case Action.MOVE:
                for key in arguments:
                    index, item = self.get(key)

                    if item is None:
                        errmsg = f'{action.name}: key "{key.data}" not found'
                        raise KeyValuesPreprocessorError(errmsg, key)

                    self.pop(index)
                    self.append(item)

            case _ as unreachable:
                assert_never(unreachable)

        first = token
        last = first if not arguments else arguments[-1]
        # Get the actual last comment (see `Action.from_comments`).
        assert last.next is not None
        last = last.next
        # Remove action comments from the output.
        if first.prev is not None:
            first.prev.next = last.next
        if last.next is not None:
            last.next.prev = first.prev

    def __iter__(self) -> Iterator[Token]:
        def yield_token(token: Token) -> Iterator[Token]:
            yield token
            yield from yieldspace(token.inext())

        if self.open is not None:
            yield from yield_token(self.open)

        for item in self.items:
            if item is None:
                continue

            yield from yield_token(item.key)

            if isinstance(item.value, MergedKeyValues):
                yield from item.value
            else:
                yield from yield_token(item.value)

            if item.condition is not None:
                yield from yield_token(item.condition)

        if self.close is not None:
            yield from yield_token(self.close)


def chain(tokens: Iterable[Token]) -> Iterator[Token]:
    tokens = iter(tokens)
    prev = next(tokens)

    for token in tokens:
        token.prev = prev
        prev.next = token

        yield prev
        prev = token

    yield prev


@parser_decorator
def merge(tokens: Iterable[Token], _depth: int) -> Iterator[Token]:
    tokens = chain(tokens)

    # Peek first token.
    tokens = iter(tokens)
    first = next(tokens)
    tokens = itertools.chain([first], tokens)

    merged = MergedKeyValues()
    merged.parse(tokens)

    yield first

    if first.tag is not TokenTag.EOF:
        yield from yieldspace(first.inext())

    yield from merged
