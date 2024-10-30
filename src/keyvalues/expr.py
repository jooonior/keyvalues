from __future__ import annotations

import itertools
import math
import operator
import re
from decimal import Decimal
from typing import TYPE_CHECKING, TypedDict, overload

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Iterator,
        Sequence,
    )
    from typing import (
        Any,
        Required,
        Self,
        TypeVar,
        Unpack,
    )

from . import utils
from .parse import ParsedTokenTag
from .token import Token, TokenError

if TYPE_CHECKING:
    from .parse import ParsedToken


class ExpressionTokenTag(utils.RegexEnum):
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

    AT  = r"@"

    OPEN   = r"\("
    CLOSE  = r"\)"

    # Sign only relevant with `re.fullmatch`, since PLUS/MINUS match first.
    HEX    = r"[+-]?0[xX][0-9a-fA-F]+"
    NUMBER = r"[+-]?(?:\.\d+|\d+(?:\.\d*)?)"

    PATH   = r"\.[/.\w]*"
    NAME   = r"[^\W\d][\w.]*"

    SPACE  = r"\s+"
    ERROR  = r".+?"  # only single character, unless matching whole string
    EOF    = r"$"

    UNSET  = ()
    # fmt: on


class ExpressionNumber(utils.RegexEnum):
    """Member names match `ExpressionTokenTag`."""

    # fmt: off
    HEX    = ExpressionTokenTag.HEX.pattern
    NUMBER = ExpressionTokenTag.NUMBER.pattern
    # fmt: off


class ExpressionTokenMeta(TypedDict, total=False):
    pass


class ExpressionToken(Token[ExpressionTokenTag, ExpressionTokenMeta]):
    @classmethod
    def from_parsed(
        cls,
        token: ParsedToken,
    ) -> Self:
        if match := ExpressionTokenTag.fullmatch(token.data):
            assert match.lastindex is not None
            tag = ExpressionTokenTag(match.lastindex)
        else:
            tag = ExpressionTokenTag.UNSET

        return cls.from_other(token, tag=tag)


class ExpressionError(TokenError):
    pass


class UnclosedParenthesesError(ExpressionError):
    def __init__(
        self,
        opening: ExpressionToken,
        closing: ExpressionToken,
    ):
        super().__init__("unclosed (", opening)
        self.__cause__ = ExpressionError("expected )", closing)


def tokenize(expression: Token[Any, Any]) -> Iterator[ExpressionToken]:
    for match in ExpressionTokenTag.finditer(expression.data):
        assert match.lastindex is not None
        tag = ExpressionTokenTag(match.lastindex)

        if tag is ExpressionTokenTag.SPACE:
            continue

        token = ExpressionToken.from_other(
            expression[match.start() : match.end()],
            tag=tag,
        )

        if tag is ExpressionTokenTag.ERROR:
            errmsg = f'unexpected character "{token.data}" in expression'
            raise ExpressionError(errmsg, token)

        yield token


if TYPE_CHECKING:
    Ret = TypeVar("Ret", int, float, Decimal, str)
    T = TypeVar("T", int, str, Decimal)

    class OperatorKwArgs(TypedDict, total=False):
        bp: Required[int]
        infixl: bool


class Operator:
    def __init__(
        self,
        op: Callable[..., Ret],
        *,
        bp: int,
        infixl: bool = True,
        cast: Callable[[Decimal], Any] | None = None,
    ):
        if cast is None:

            def convert(token: ExpressionToken) -> Any:
                return token_to_decimal(token)

        else:

            def convert(token: ExpressionToken) -> Any:
                return cast(token_to_decimal(token))

        def wrapped(*tokens: ExpressionToken) -> Decimal:
            return Decimal(op(*map(convert, tokens)))

        self.op = wrapped
        self.lbp = bp * 2 + (1 - infixl)
        self.rbp = bp * 2

    def _invoke(
        self,
        token: ExpressionToken,
        *args: ExpressionToken,
    ) -> ExpressionToken:
        try:
            result = self.op(*args)
        except Exception as exc:
            errmsg = "error evaluating expression"
            raise ExpressionError(errmsg, token) from exc

        return token.clone(
            data=format(result, "f"),
            tag=ExpressionTokenTag.NUMBER,
        )


class UnaryOperator(Operator):
    @overload
    def __init__(
        self,
        op: Callable[[Decimal], Ret],
        **kwargs: Unpack[OperatorKwArgs],
    ): ...

    @overload
    def __init__(
        self,
        op: Callable[[T], Ret],
        *,
        cast: Callable[[Decimal], T],
        **kwargs: Unpack[OperatorKwArgs],
    ): ...

    def __init__(
        self,
        op: Callable[[T], Ret],
        *,
        cast: Callable[[Decimal], T] | None = None,
        **kwargs: Unpack[OperatorKwArgs],
    ):
        super().__init__(op, **kwargs, cast=cast)

    def invoke(
        self,
        token: ExpressionToken,
        arg: ExpressionToken,
    ) -> ExpressionToken:
        return self._invoke(token, arg)


class BinaryOperator(Operator):
    @overload
    def __init__(
        self,
        op: Callable[[Decimal, Decimal], Ret],
        **kwargs: Unpack[OperatorKwArgs],
    ): ...

    @overload
    def __init__(
        self,
        op: Callable[[T, T], Ret],
        *,
        cast: Callable[[Decimal], T],
        **kwargs: Unpack[OperatorKwArgs],
    ): ...

    def __init__(
        self,
        op: Callable[[T, T], Ret],
        *,
        cast: Callable[[Decimal], T] | None = None,
        **kwargs: Unpack[OperatorKwArgs],
    ):
        super().__init__(op, **kwargs, cast=cast)

    def invoke(
        self,
        token: ExpressionToken,
        left: ExpressionToken,
        right: ExpressionToken,
    ) -> ExpressionToken:
        return self._invoke(token, left, right)


def token_to_decimal(token: ExpressionToken) -> Decimal:
    match token.tag:
        case ExpressionTokenTag.NUMBER:
            return Decimal(token.data)

        case ExpressionTokenTag.HEX:
            return Decimal(int(token.data, base=16))

        case _:
            errmsg = f'"{token.data}" cannot be converted to a number'
            raise ExpressionError(errmsg, token)


def token_to_int(token: ExpressionToken) -> int:
    return int(token_to_decimal(token))


PREFIX_OPERATORS = {
    ExpressionTokenTag.PLUS: UnaryOperator(operator.pos, bp=9),
    ExpressionTokenTag.MINUS: UnaryOperator(operator.neg, bp=9),
    ExpressionTokenTag.NOT: UnaryOperator(operator.not_, bp=9),
    ExpressionTokenTag.BNOT: UnaryOperator(operator.inv, bp=9, cast=int),
}


INFIX_OPERATORS = {
    ExpressionTokenTag.AND: BinaryOperator(lambda x, y: x and y, bp=0),
    ExpressionTokenTag.OR: BinaryOperator(lambda x, y: x or y, bp=1),
    ExpressionTokenTag.EQ: BinaryOperator(operator.eq, bp=2),
    ExpressionTokenTag.NE: BinaryOperator(operator.ne, bp=2),
    ExpressionTokenTag.LE: BinaryOperator(operator.le, bp=3),
    ExpressionTokenTag.LT: BinaryOperator(operator.lt, bp=3),
    ExpressionTokenTag.GE: BinaryOperator(operator.ge, bp=3),
    ExpressionTokenTag.GT: BinaryOperator(operator.gt, bp=3),
    ExpressionTokenTag.BAND: BinaryOperator(operator.and_, bp=4, cast=int),
    ExpressionTokenTag.BOR: BinaryOperator(operator.or_, bp=5, cast=int),
    ExpressionTokenTag.LSHIFT: BinaryOperator(operator.lshift, bp=6, cast=int),
    ExpressionTokenTag.RSHIFT: BinaryOperator(operator.rshift, bp=6, cast=int),
    ExpressionTokenTag.PLUS: BinaryOperator(operator.add, bp=7),
    ExpressionTokenTag.MINUS: BinaryOperator(operator.sub, bp=7),
    ExpressionTokenTag.MULT: BinaryOperator(operator.mul, bp=8),
    ExpressionTokenTag.DIVIDE: BinaryOperator(operator.truediv, bp=8),
    ExpressionTokenTag.MODULO: BinaryOperator(operator.mod, bp=8),
    ExpressionTokenTag.POWER: BinaryOperator(operator.pow, bp=9, infixl=False),
}


if TYPE_CHECKING:
    ExpanderFn = Callable[
        [ExpressionToken, Sequence[ExpressionToken] | None],
        ParsedToken,
    ]


def pratt(
    tokens: utils.Peekable[ExpressionToken],
    expander: ExpanderFn,
    *,
    min_bp: float = 0,
) -> ExpressionToken:
    token = next(tokens)

    match token.tag:
        case ExpressionTokenTag.NUMBER | ExpressionTokenTag.HEX:
            lhs = token

        case ExpressionTokenTag.OPEN:
            lhs = pratt(tokens, expander, min_bp=0)

            close = next(tokens)
            if close.tag is not ExpressionTokenTag.CLOSE:
                raise UnclosedParenthesesError(token, close)

        case ExpressionTokenTag.NAME:
            """
            while definition.arity == 0:
                name = expander(evaluate(definition.body), [])
                if name.tag != NAME and name.tag != PATH:
                    break
                definition = defs.get(name)
            ---
            while ...:
                try:
                    expanded = expander(token, [])
                except ArgumentError:
                    pass
            """
            arguments = []

            while True:
                match tokens[0].tag:
                    case ExpressionTokenTag.OPEN:
                        paren = next(tokens)
                        arguments.append(pratt(tokens, expander, min_bp=0))

                        close = next(tokens)
                        if close.tag is not ExpressionTokenTag.CLOSE:
                            raise UnclosedParenthesesError(paren, close)

                    case ExpressionTokenTag.NUMBER | ExpressionTokenTag.HEX:
                        arguments.append(next(tokens))

                    case ExpressionTokenTag.NAME:
                        expanded = expander(next(tokens), [])
                        arguments.append(ExpressionToken.from_parsed(expanded))

                    case ExpressionTokenTag.PATH:
                        expanded = expander(next(tokens), None)
                        arguments.append(ExpressionToken.from_parsed(expanded))

                    case _:
                        break

            lhs = ExpressionToken.from_parsed(expander(token, arguments))

        case ExpressionTokenTag.PATH:
            lhs = ExpressionToken.from_parsed(expander(token, None))

        case ExpressionTokenTag.EOF:
            errmsg = "unexpected end of expression"
            raise ExpressionError(errmsg, token)

        case _:
            prefix_op = PREFIX_OPERATORS.get(token.tag)
            if prefix_op is None:
                errmsg = f'unexpected token "{token.data}" in expression'
                raise ExpressionError(errmsg, token)

            rhs = pratt(tokens, expander, min_bp=prefix_op.rbp)
            lhs = prefix_op.invoke(token, rhs)

    while True:
        token = tokens[0]

        match token.tag:
            case ExpressionTokenTag.EOF:
                break

            case ExpressionTokenTag.CLOSE:
                break

            # Parse infix operators.

            case ExpressionTokenTag.AT:
                next(tokens)

                rhs = pratt(tokens, expander, min_bp=math.inf)
                index = token_to_int(rhs)
                if index <= 0:
                    errmsg = "index starts from 1"
                    raise ExpressionError(errmsg, rhs)

                matches = ExpressionNumber.finditer(lhs.data)
                match = next(itertools.islice(matches, index - 1, None), None)
                if match is None:
                    errmsg = "index out of range"
                    raise ExpressionError(errmsg, rhs)

                assert match.lastgroup is not None
                lhs = ExpressionToken.from_match(
                    match,
                    # Match group indices are different, but names match.
                    tag=ExpressionTokenTag[match.lastgroup],
                )

            case _:
                infix_op = INFIX_OPERATORS.get(token.tag)
                if infix_op is None:
                    errmsg = f'expected infix operator, got "{token.data}"'
                    raise ExpressionError(errmsg, token)

                if infix_op.lbp <= min_bp:
                    break

                next(tokens)

                rhs = pratt(tokens, expander, min_bp=infix_op.rbp)
                lhs = infix_op.invoke(token, lhs, rhs)

    return lhs


def operator_at(lhs: ExpressionToken, rhs: ExpressionToken) -> ExpressionToken:
    index = token_to_int(rhs)
    sign = 0

    for token in tokenize(lhs):
        match token.tag:
            case ExpressionTokenTag.EOF:
                break

            # Parse leading "+" as part of the number.
            case ExpressionTokenTag.PLUS if sign == 0:
                sign = 1

            # Parse leading "-" as part of the number.
            case ExpressionTokenTag.MINUS if sign == 0:
                sign = -1

            case ExpressionTokenTag.NUMBER | ExpressionTokenTag.HEX:
                # We want 1-based indexing.
                if index == 1:
                    if sign == -1:
                        token.data = f"-{token.data}"

                    return token

                sign = 0
                index -= 1

            # Ignore all other tokens.
            case _:
                sign = 0

    errmsg = "index out of range"
    raise ExpressionError(errmsg, rhs)


def evaluate(
    token: ParsedToken,
    expander: ExpanderFn,
) -> ParsedToken:
    tokens = utils.Peekable(tokenize(token))

    result = pratt(tokens, expander)
    return token.clone(
        data=result.data,
        tag=ParsedTokenTag.QUOTED,
    )


def expand(
    token: ParsedToken,
    expander: ExpanderFn,
) -> ParsedToken:
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
            parts.append(evaluate(token[start:cursor], expander).data)
            start = cursor

    parts.append(token.data[start:])
    return token.clone(data="".join(parts), tag=ParsedTokenTag.QUOTED)
