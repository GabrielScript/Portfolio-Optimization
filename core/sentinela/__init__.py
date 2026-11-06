"""
Sentinela: a DeepSeek watch agent over the Composite's names (spec
``docs/superpowers/specs/2026-11-05-sentinela-design.md``, DECISOES 262-272). Code reads the official sources and
turns them into items; the model is called only for a name with new items; every answer is checked against the
items' verbatim text; actions happen only in the fictitious "Composite + Sentinela" account.
"""
