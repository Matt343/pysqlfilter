import pyparsing as pp
import abc
import dataclasses
from typing import Optional

class TypecheckError(Exception):
    def __init__(self, *args):
        super().__init__(*args)

@dataclasses.dataclass
class ValueType:
    name: str
    aliases: list[str] = None
    is_integer: bool = False
    is_numeric: bool = False
    is_text: bool = False
    is_literal: bool = False
    is_any: bool = False

    def assignable(self, other: 'ValueType') -> bool:
        if self.is_any or other.is_any:
            return True
        if other.name == self.name:
            return True
        if self.aliases and other.name in self.aliases:
            return True
        if other.aliases and self.name in other.aliases:
            return True
        if self.is_integer and other.is_numeric:
            return True
        if (self.is_literal or other.is_literal) and self.is_integer == other.is_integer:
            return True
        if (self.is_literal or other.is_literal) and self.is_numeric == other.is_numeric:
            return True
        if (self.is_literal or other.is_literal) and self.is_text == other.is_text:
            return True
        
    def resolve(self, other: 'ValueType') -> 'ValueType':
        if self.is_any:
            return other
        if other.is_any:
            return other
        if self.is_integer and other.is_numeric:
            return other
        if self.is_literal:
            return other
        return self

@dataclasses.dataclass
class ListType(ValueType):
    elem: Optional[ValueType] = None

    @staticmethod
    def unknown():
        return ListType(name='List<unknown>', elem=None)

    @staticmethod
    def of(e: ValueType):
        return ListType(name=f'List<{e.name}>', elem=e)
    
@dataclasses.dataclass
class RangeType(ValueType):
    elem: Optional[ValueType] = None

    @staticmethod
    def of(e: ValueType):
        return RangeType(name=f'Range<{e.name}>', elem=e)
    

integer_type = ValueType(name="integer", aliases=["int", "bigint", "smalling", "int2", "int4", "int8"], is_numeric=True, is_integer=True)
boolean_type = ValueType(name="boolean", aliases=["bool"])
float_type = ValueType(name="double precision", aliases=["real", "float4", "float8"], is_numeric=True)
string_type = ValueType(name="varchar", aliases=["text", "character varying"], is_text=True)
null_type = ValueType(name="null", is_literal=True, is_any=True)

default_types = (
    integer_type,
    boolean_type,
    ValueType(name="date"),
    float_type,
    ValueType(name="interval"),
    ValueType(name="json"),
    ValueType(name="jsonb"),
    ValueType(name="numeric", aliases=["decimal"], is_numeric=True),
    ValueType(name="time", aliases=["timetz"]),
    ValueType(name="timestamp", aliases=["timestamptz"]),
    string_type,
)

class AstNode:
    @abc.abstractmethod
    def sql(self) -> str:
        pass

    @abc.abstractmethod
    def typecheck(self, columns: dict[str, ValueType], expected_type: ValueType = None) -> ValueType:
        pass

class BinaryOp(AstNode):
    def __init__(self, parsed: pp.ParseResults):
        try:
            p = parsed.as_list()[0]
            self.operator: AstNode = p[1]
            self.operands: tuple[AstNode, AstNode] = p[0], p[2]
        except Exception as e:
            print("failed constructor")
            print(parsed)
            print(e)
            raise

    def __repr__(self) -> str:
        return f'BinaryOp({self.operands[0]!r} {self.operator!r} {self.operands[1]!r})'

    def sql(self) -> str:
        return f"({self.operands[0].sql()} {self.operator.sql()} {self.operands[1].sql()})"
    
    def typecheck(self, columns: dict[str, ValueType], expected_type: ValueType = None) -> ValueType:
        if self.operator.op == 'in':
            op1 = self.operands[0].typecheck(columns)
            op2 = self.operands[1].typecheck(columns, expected_type=ListType.of(op1))
            if not isinstance(op2, ListType):
                raise TypecheckError(f"expected list of values, got {op2.name}")
            op2 = op2.elem
            if not op1.assignable(op2) and not op2.assignable(op1):
                raise TypecheckError(f"operator {self.operator.sql()} must be applied between matching types, got {op1.name} and {op2.name}")
            result = boolean_type
            if expected_type and not expected_type.assignable(result):
                raise TypecheckError(f"expected {expected_type.name}, got {result.name}")
            return result
        elif self.operator.op == 'between':
            op1 = self.operands[0].typecheck(columns)
            op2 = self.operands[1].typecheck(columns, expected_type=RangeType.of(op1))
            if not isinstance(op2, RangeType):
                raise TypecheckError(f"expected range value, got {op2.name}")
            op2 = op2.elem
            if not op1.assignable(op2) and not op2.assignable(op1):
                raise TypecheckError(f"operator {self.operator.sql()} must be applied between matching types, got {op1.name} and {op2.name}")
            result = boolean_type
            if expected_type and not expected_type.assignable(result):
                raise TypecheckError(f"expected {expected_type.name}, got {result.name}")
            return result
        elif self.operator.op == 'and':
            op1 = self.operands[0].typecheck(columns)
            op2 = self.operands[1].typecheck(columns, expected_type=op1)
            if op1.assignable(op2):
                result = op1.resolve(op2)
            elif op2.assignable(op1):
                result = op2.resolve(op1)
            else:
                raise TypecheckError(f"operator {self.operator.sql()} must be applied between matching types, got {op1.name} and {op2.name}")
            if expected_type and isinstance(expected_type, RangeType):
                return RangeType.of(result)
            if not result.assignable(boolean_type):
                raise TypecheckError(f"operator {self.operator.sql()} must be applied to boolean values, got {op1.name} and {op2.name}")
            return result
        elif self.operator.op in ('<', '>', '<=', '>=', '<>', '=', 'is', 'is not'):
            op1 = self.operands[0].typecheck(columns)
            op2 = self.operands[1].typecheck(columns, expected_type=op1)
            if not op1.assignable(op2) and not op2.assignable(op1):
                raise TypecheckError(f"operator {self.operator.sql()} must be applied between matching types, got {op1.name} and {op2.name}")
            return boolean_type
        else:
            op1 = self.operands[0].typecheck(columns)
            op2 = self.operands[1].typecheck(columns, expected_type=op1)
            if op1.assignable(op2):
                result = op1.resolve(op2)
            elif op2.assignable(op1):
                result = op2.resolve(op1)
            else:
                raise TypecheckError(f"operator {self.operator.sql()} must be applied between matching types, got {op1.name} and {op2.name}")
            
            if expected_type and not expected_type.assignable(result):
                raise TypecheckError(f"expected {expected_type.name}, got {result.name}")
            return result
            

class UnaryOp(AstNode):
    def __init__(self, parsed: pp.ParseResults):
        try:
            p = parsed.as_list()[0]
            self.operator = p[0]
            self.operand = p[1]
        except Exception as e:
            print("failed constructor")
            print(parsed)
            print(e)
            raise

    def __repr__(self) -> str:
        return f'UnaryOp({self.operator!r} {self.operand!r})'
    
    def sql(self) -> str:
        return f'({self.operator.sql()} {self.operand.sql()})'
    
    def typecheck(self, columns: dict[str, ValueType], expected_type: ValueType = None) -> ValueType:
        if self.operator.op == 'not':
            op = self.operand.typecheck(columns, expected_type=boolean_type)
            result = op.resolve(boolean_type)
            if expected_type and not result.assignable(expected_type):
                raise TypecheckError(f"expected {expected_type.name}, got {result.name}")
            return result
        else:
            numeric_lit_type = ValueType(name='numeric', is_literal=True, is_numeric=True)
            op = self.operand.typecheck(columns, expected_type=numeric_lit_type)
            result = op.resolve(numeric_lit_type)
            if expected_type and not expected_type.assignable(result):
                raise TypecheckError(f"expected {expected_type.name}, got {result.name}")
            return result

class Value(AstNode):
    def __init__(self, parsed: pp.ParseResults, typ: Optional[ValueType] = None):
        try:
            self.value = parsed.as_list()[0]
            self.typ = typ or null_type
        except Exception as e:
            print("failed constructor")
            print(parsed)
            print(e)
            raise

    def __repr__(self) -> str:
        return f'Value({self.value!r}, {self.typ})'

    def sql(self) -> str:
        return self.value

    @staticmethod
    def Typed(typ: Optional[ValueType]):
        return lambda parsed: Value(parsed, typ)
    
    def typecheck(self, columns: dict[str, ValueType], expected_type: ValueType = None) -> ValueType:
        if expected_type:
            result = self.typ.resolve(expected_type)
        else:
            result = self.typ
        if expected_type and not expected_type.assignable(result):
            raise TypecheckError(f"expected {expected_type.name}, got {result.name}")
        return result
    
class Ident(Value):
    def __init__(self, parsed: pp.ParseResults):
        try:
            self.name = ''
            for p in parsed.as_list():
                if isinstance(p, (Ident, Operator)):
                    self.name += p.sql()
                else:
                    self.name += p
        except Exception as e:
            print("failed constructor")
            print(parsed)
            print(e)
            raise

    def __repr__(self) -> str:
        return f'Ident({self.name})'

    def sql(self) -> str:
        return f'{self.name}'
    
    def typecheck(self, columns: dict[str, ValueType], expected_type: ValueType = None) -> ValueType:
        if self.name not in columns:
            raise TypecheckError(f"unknown identifier {self.name}")
        result = columns[self.name]

        if expected_type and not expected_type.assignable(result):
            raise TypecheckError(f"expected {expected_type.name}, got {result.name}")
        return result

    
class ListLiteral(Value):
    def __init__(self, parsed: pp.ParseResults):
        try:
            self.values: list[AstNode] = parsed[1:-1]
            self.typ: ValueType = ListType.unknown()
        except Exception as e:
            print("failed constructor")
            print(parsed)
            print(e)
            raise

    def __repr__(self):
        return f'ListLiteral({self.values!r})'
    
    def sql(self) -> str:
        return f'({", ".join(map(self.values, lambda v: v.sql()))})'

    def typecheck(self, columns: dict[str, ValueType], expected_type: ValueType = None) -> ValueType:
        if expected_type:
            if not isinstance(expected_type, ListType):
                raise TypecheckError(f"expected {expected_type}, got list of values")
            elem_type = expected_type.elem
        else:
            elem_type = None

        for v in self.values:
            result = v.typecheck(columns, expected_type=elem_type)
            if elem_type and not elem_type.assignable(result):
                raise TypecheckError(f"expected {elem_type.name}, got {result.name}")
            if not result.is_any:
                elem_type = result
        if elem_type:
            return ListType.of(elem_type)
        else:
            return ListType.unknown()

class Operator(AstNode):
    def __init__(self, parsed: pp.ParseResults):
        try:
            self.op = " ".join(parsed.as_list())
        except Exception as e:
            print("failed constructor")
            print(parsed)
            print(e)
            raise

    def __repr__(self):
        return f'Operator({self.op})'

    def sql(self) -> str:
        return self.op
    
    def typecheck(self, columns: dict[str, ValueType], expected_type: ValueType = None) -> ValueType:
        return ValueType(name='operator')

# based on Postgres reserved words list
default_reserved_keywords = ("all", "analyse", "analyze", "and", "any", "asc", "asymmetric", "authorization", "between", "binary", "both", "case", "cast", "check", "collate", "collation", "column", "concurrently", "constraint", "cross", "current_catalog", "current_date", "current_role", "current_schema", "current_time", "current_timestamp", "current_user", "default", "deferrable", "desc", "distinct", "do", "else", "end", "false", "foreign", "full", "ilike", "in", "initially", "inner", "is", "join", "lateral", "leading", "left", "like", "localtime", "localtimestamp", "natural", "not", "notnull", "null", "only", "or", "outer", "placing", "primary", "references", "right", "select", "session_user", "similar", "some", "symmetric", "system_user", "table", "tablesample", "then", "trailing", "true", "user", "using", "variadic", "verbose", "when")


class FilterParser:
    def __init__(self, reserved_keywords=None, types=None):
        self.reserved_keywords = reserved_keywords or default_reserved_keywords
        self.types = types or default_types

        identifier = (pp.dbl_quoted_string() | pp.Word(pp.alphas, pp.identbodychars).add_condition(lambda toks: toks[0].lower() not in self.reserved_keywords, message="reserved keyword")).add_parse_action(Ident)
        string_lit = pp.sgl_quoted_string().set_parse_action(Value.Typed(ValueType(name="text", is_literal=True, is_text=True)))
        int_lit = pp.common.signed_integer.set_parse_action(Value.Typed(ValueType(name="integer", is_literal=True, is_numeric=True, is_integer=True)))
        number_lit = pp.common.real.set_parse_action(Value.Typed(ValueType(name="number", is_literal=True, is_numeric=True)))
        bool_lit = pp.one_of("true false", caseless=True, as_keyword=True).set_parse_action(Value.Typed(ValueType(name="boolean", is_literal=True)))
        null_lit = pp.CaselessKeyword("null").set_parse_action(Value.Typed(null_type))
        dot_op = pp.Literal('.').set_parse_action(Operator)
        reference = (identifier + dot_op + identifier).set_parse_action(Ident)
        value = string_lit | number_lit | int_lit | bool_lit | null_lit | reference | identifier
        list_lit = (pp.Literal("(") + pp.DelimitedList(value, allow_trailing_delim=True) + pp.Literal(")")).set_parse_action(ListLiteral)

        term = list_lit | value

        negation_op = pp.Literal('-').set_parse_action(Operator)
        not_op = pp.CaselessKeyword('not').set_parse_action(Operator)
        is_op = (pp.CaselessKeyword("is") + pp.Opt(pp.CaselessKeyword("not"))).set_parse_action(Operator)
        inequality_op = pp.one_of("> <").set_parse_action(Operator)
        comparison_op = pp.one_of(">= <= <>").set_parse_action(Operator)
        equality_op = pp.Literal("=").set_parse_action(Operator)
        in_op = pp.CaselessKeyword("in").set_parse_action(Operator)
        multiplication_op = pp.one_of("* / %").set_parse_action(Operator)
        addition_op = pp.one_of("+ -").set_parse_action(Operator)
        and_op = pp.CaselessKeyword("and").set_parse_action(Operator)
        or_op = pp.CaselessKeyword("or").set_parse_action(Operator)
        cast_op = pp.Literal("::").set_parse_action(Operator)
        between_op = pp.CaselessKeyword('between').set_parse_action(Operator)
        like_op = pp.one_of('like ilike', caseless=True, as_keyword=True).set_parse_action(Operator)

        # precedence and associativity set based on Postgres operators
        bin_ops = pp.infix_notation(term, [
            (cast_op, 2, pp.opAssoc.LEFT, BinaryOp),
            (negation_op, 1, pp.opAssoc.RIGHT, UnaryOp),
            (multiplication_op, 2, pp.opAssoc.LEFT, BinaryOp),
            (addition_op, 2, pp.opAssoc.LEFT, BinaryOp),
            (is_op, 2, pp.opAssoc.LEFT, BinaryOp),
            (comparison_op, 2, pp.OpAssoc.LEFT, BinaryOp),
            (in_op, 2, pp.opAssoc.LEFT, BinaryOp),
            (between_op, 2, pp.opAssoc.LEFT, BinaryOp),
            (like_op, 2, pp.opAssoc.LEFT, BinaryOp),
            (inequality_op, 2, pp.opAssoc.LEFT, BinaryOp),
            (equality_op, 2, pp.opAssoc.RIGHT, BinaryOp),
            (not_op, 1, pp.opAssoc.RIGHT, UnaryOp),
            (and_op, 2, pp.opAssoc.LEFT, BinaryOp),
            (or_op, 2, pp.opAssoc.LEFT, BinaryOp),
        ])

        self._parser = bin_ops
        self._parser.enable_packrat()

    def resolve_types(self, columns: dict[str, str]) -> dict[str, ValueType]:
        out = {}
        for c, t in columns.items():
            found = False
            for typ in self.types:
                if typ.name == t:
                    out[c] = typ
                    found = True
                elif typ.aliases and t in typ.aliases:
                    out[c] = typ
                    found = True
            if not found:
                raise TypecheckError(f"unknown type name {t}")
        return out

    def parse(self, filter: str) -> AstNode:
        return self._parser.parse_string(filter, parse_all=True)[0]
    
    def build_sql(self, filter: str, columns: dict[str, str] = None):
        parse_result = self.parse(filter)
        if columns:
            typemap = self.resolve_types(columns)
            parse_result.typecheck(typemap, expected_type=boolean_type)
        return parse_result.sql()

if __name__ == "__main__": 
    print("--- Running parser tests ---")
    parser = FilterParser()
    outcome, parse_output = parser._parser.run_tests(
        tests=[
            "a",
            "'a'",
            "-1.2",
            "true",
            "null",
            "a::varchar",
            "not a",
            "a.b is 'b'",
            "'a' + 'b'",
            "(a, b, c)",
            "a in ('a', b)",
            "\"in\" + b > 1",
            "foo like 'bar%'",
            "(foo > bar) and not baz",
            "a is not null",
            "1 + 1 > 2",
            "(foo > bar) = (not a)",
            "1 + a = c and not foo <> bar",
        ],
    )
    assert outcome, "Failed parser tests!"

    print("--- Running parser negative tests ---")
    outcome, parse_output = parser._parser.run_tests(
        tests = [
            "random + in foo",
            "AND > foo",
        ],
        failure_tests=True
    )
    assert outcome, "Failed parser negative tests!"

    print("--- testing sql builder ---")
    out = parser.build_sql('foo + \'lit\' > 1 and not "bar baz"')
    assert out == '(((foo + \'lit\') > 1) and (not "bar baz"))', out

    columns = {'s': 'text', 'b': 'bool', 'i': 'int', 'f': 'real'}
    out = parser.build_sql('s + \'lit\' > 1 and not b', columns=columns)
    assert out == '(((s + \'lit\') > 1) and (not b))', out
    out = parser.build_sql('1 + i / f > -0.1', columns=columns)
    assert out == '((1 + (i / f)) > (- 0.1))', out

    print("----- Passed -----")
 