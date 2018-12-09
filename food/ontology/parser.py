

from lark import Lark, Transformer
from multidict import MultiDict


grammar = '''

    start: line*
    
    line: NEWLINE
         | COMMENT
         | definition
    
    definition: header fact*
    
    header: identifier (L_BRACE parameters R_BRACE)? (COLON parents)? NEWLINE
    
    parameters: parameter (COMMA parameter)*
    parameter: identifier (COLON identifier)? (EQUAL identifier)?
    
    parents: parent (COMMA parent)*
    parent: identifier (L_BRACE identifiers R_BRACE)?
    
    fact: INDENT identifier COLON value NEWLINE
    
    identifiers: identifier (COMMA identifier)*
    identifier: /[a-zA-Z_][a-zA-Z_0-9]*/
    
    value: /[a-zA-Z_0-9]+/
    
    NEWLINE: /\ */ _NEWLINE
    L_BRACE: /\ *\(\ */
    R_BRACE: /\ *\)\ */
    COMMA: /\ *,\ */
    COLON: /\ *\:\ */
    EQUAL: /\ *=\ */
    COMMENT: /#.*/ _NEWLINE
    INDENT: /\ \ /
    
    %import common.NEWLINE -> _NEWLINE
    
'''


class Definition:
    def __init__(self, identifier, parameters, parents, facts):
        self.identifier = identifier
        self.parameters = parameters
        self.parents = parents
        self.facts = facts
    
    def __repr__(self):
        result = self.identifier.value
        if len(self.parameters) > 0:
            result = f'{result}({", ".join(repr(p) for p in self.parameters)})'
        if len(self.parents) > 0:
            result = f'{result}: {", ".join(repr(p) for p in self.parents)}'
        # TODO facts
        return result


class Parameter:
    def __init__(self, name, type, default):
        self.name = name
        self.type = type
        self.default = default
    
    def __repr__(self):
        result = self.name.value
        if self.type is not None:
            result += ':' + self.type.value
        if self.default is not None:
            result += '=' + self.default.value
        return result


class Parent:
    def __init__(self, name, values):
        self.name = name
        self.values = values
    
    def __repr__(self):
        result = self.name
        if len(self.values) > 0:
            result = f'{result}({", ".join(self.values)})'
        return result


class Identifier:
    def __init__(self, token):
        self.token = token
    
    @property
    def value(self):
        return self.token.value
    
    def __repr__(self):
        return self.value


class DefinitionTransformer(Transformer):
    
    def start(self, children):
        return [child for child in children if type(child) is Definition]
    
    def line(self, children):
        return children[0]
    
    def definition(self, children):
        (identifier, parameters, parents), *facts = children
        return Definition(identifier, parameters, parents, facts)
    
    def header(self, children):
        identifier = children[0]
        parameters = []
        parents = []
        if len(children) == 7:
            parameters = children[2]
            parents = children[5]
        elif len(children) == 5:
            parameters = children[2]
        elif len(children) == 4:
            parents = children[2]
        return identifier, parameters, parents
    
    def parameters(self, children):
        return children[::2]
    
    def parameter(self, children):
        name = children[0]
        type = None
        default = None
        if len(children) == 5:
            type = children[2]
            default = children[4]
        elif len(children) == 3:
            if children[1].type == 'COLON':
                type = children[2]
            else:
                default = children[2]
        return Parameter(name, type, default)
    
    def parents(self, children):
        return children[::2]
    
    def parent(self, children):
        # TODO parent
        name = children[0]
        if len(children) > 1:
            values = children[2::2]
        else:
            values = []
        return Parent(name, values)
    
    def fact(self, children):
        # TODO fact
        return children
    
    def identifiers(self, children):
        return children[::2]
    
    def identifier(self, children):
        token = children[0]
        return Identifier(token)
    
    def value(self, children):
        # TODO
        return children
    

parser = Lark(grammar, parser='lalr', transformer=DefinitionTransformer())


# TODO resolve identifiers
