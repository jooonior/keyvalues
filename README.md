# keyvalues

Python module for manipulating Valve's [KeyValues][keyvalues], consisting of:

- modular and extensible (and over-engineered) parser; on top of which are
  implemented
- formatter; and
- preprocessor using a superset of the standard KeyValues syntax.


## Notice

I wrote the preprocessor specifically for [9ui][9ui]. Feel free to use it in
your projects, but don't expect any support. Other than this README, there is
no documentation, but you can learn by example from the [9ui][9ui] sources.


## Preprocessor

Specific functionality is subject to change at my whim. Still, the generic
principles are outlined below.


### Directives

If a opening brace is found in place of a key, all tokens between it and a
matching balanced closing brace are interpreted as a _directive_. Directives,
including their surrounding braces, and in some cases also following tokens,
are removed from the output. They can modify the preprocessor state and omit
tokens into the output.


### Modifiers

If a token matching `$[...]` is found between a key and a value, it is removed
from output and the `...` are interpreted as a comma-separated sequence of
_flags_, which modify how the following value is processed.


### Expressions

If the `EXPAND` flag is set, any substring of a key or a value matching `$(...)`
is parsed as an _expression_ and substituted for its evaluated value.


## Known Issues

- Preprocessor does not remove whitespace around directives.


[keyvalues]: https://developer.valvesoftware.com/wiki/KeyValues
[9ui]: https://github.com/jooonior/9ui
