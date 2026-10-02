"""Read-only BDDL S-expression parsing used by exact target-region masks."""
import re


def parse_bddl(text):
    tokens = re.findall(r"\(|\)|[^\s()]+", re.sub(r";[^\n]*", "", text))
    stack, result = [], None
    for token in tokens:
        if token == "(":
            value = []
            if stack:
                stack[-1].append(value)
            stack.append(value)
        elif token == ")":
            if not stack:
                raise ValueError("unbalanced BDDL")
            result = stack.pop()
        else:
            if not stack:
                raise ValueError("BDDL token outside expression")
            stack[-1].append(token.casefold())
    if stack or not isinstance(result, list) or not result or result[0] != "define":
        raise ValueError("invalid BDDL definition")
    return result


def sections(text):
    return {value[0]: value[1:] for value in parse_bddl(text)[1:] if isinstance(value, list)}


def typed_declarations(tokens):
    result, pending, index = {}, [], 0
    while index < len(tokens):
        if tokens[index] == "-":
            if not pending or index + 1 >= len(tokens):
                raise ValueError("malformed typed declaration")
            result.update({name: tokens[index + 1] for name in pending})
            pending = []
            index += 2
        else:
            pending.append(tokens[index])
            index += 1
    result.update({name: "object" for name in pending})
    return result
