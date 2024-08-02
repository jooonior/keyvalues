from __future__ import annotations

import enum
import re
from typing import TYPE_CHECKING, Any, Self


class AutoIntEnum(enum.IntEnum):
    def __new__(cls, value: int | None = None) -> Self:
        if value is None:
            value = len(cls)
        obj = int.__new__(cls, value)
        obj._value_ = value
        return obj


class AutoFlagEnum(enum.IntFlag):
    def __new__(cls, value: int | None = None) -> Self:
        if value is None:
            value = 1 << len(cls)
        obj = int.__new__(cls, value)
        obj._value_ = value
        return obj


if TYPE_CHECKING:
    # Make type checker think that `RegexEnum` has attributes of `re.Pattern`.

    class RegexEnumMeta(enum.EnumType, re.Pattern[str]):  # type: ignore[misc]
        pass

else:

    class RegexEnumMeta(enum.EnumType):
        def __getattr__(self, name: str) -> Any:
            if self is RegexEnum:
                return getattr(super(), name)

            attr = getattr(self._re, name)
            setattr(self, name, attr)
            return attr


class RegexEnum(enum.IntEnum, metaclass=RegexEnumMeta):
    _ignore_ = "pattern"

    pattern: str

    def __new__(cls, value: str) -> Self:
        index = len(cls) + 1

        obj = int.__new__(cls, index)
        obj._value_ = index
        obj.pattern = value

        return obj

    def __init_subclass__(cls) -> None:
        for member in cls:
            # Convert outermost capturing group into named capturing group.
            open_named_group = f"(?P<{member.name}>"
            regex, n = re.subn(
                r"(?<!\\)\((?!\?)", open_named_group, member.pattern, count=1
            )
            if n == 0:
                # No existing capture group, wrap whole regex in one.
                member.pattern = f"{open_named_group}{regex})"
            else:
                # Wrap regex in non-capturing group to ensure correct
                # interpretation when multiple regexes are joined by "|".
                member.pattern = f"(?:{regex})"

        joined = "|".join(member.pattern for member in cls)
        cls._re = re.compile(joined)  # type: ignore[attr-defined]
