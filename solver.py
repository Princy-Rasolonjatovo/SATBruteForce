# -*- coding: utf-8 -*-
"""
@author : Princy Rasolonjatovo
@email  : princy.m.rasolonjatovo@gmail.com
@github : princy-rasolonjatovo
"""
from __future__ import annotations
from typing import Callable, List
from itertools import product
import re



class Node:
    def __init__(self, value: object, left_child: Node, right_child: Node):
        self.value = value
        self.left = left_child
        self.right = right_child

    def __repr__(self):
        return"""({0})""".format(self.value)


class Token:
    _instances = dict() # keep track of created token
    def __new__(cls, name: str):
        instance = super(Token, cls).__new__(cls)
        instance.__init__(name)
        
        if cls._instances.get(name) is not None:
            return cls._instances.get(name)
        else:
            cls._instances[name] = instance
            return instance

    @classmethod
    def removeAll(cls):
        del cls._instances
        cls._instances = dict()   

    @classmethod
    def getTokens(cls) -> List[Token]:
        return [cls._instances[key] for key  in sorted(cls._instances.keys())]

    @classmethod
    def getInstancesCount(cls) -> int:
        return len(cls._instances.keys())

    def __init__(self, name: str):
        self.name: str = name
        self.value: bool = False

    def __repr__(self):
        return 'Token<%s>' % (self.name)
    
    def __eq__(self, other: Token):
        if not isinstance(other, Token):
            raise Exception('[ComparableError] cannot compare object<%s> with <Token> object' % (type(other)))
        return hash(self) == hash(other)

    def __hash__(self)->int:
        return hash(self.name)


# Operators objects
class Operator:
    def __init__(self, name: str, symbol: str, fn: Callable[[bool, bool], bool], priority: int, isUnary: bool= False):
        self._name = name
        self._symbol = symbol
        self._fn = fn
        self._priority = priority
        self._isUnary = isUnary

    @property
    def isUnary(self) -> bool:
        return self._isUnary

    @property
    def name(self):
        return self._name

    @property
    def priority(self):
        return self._priority

    @property
    def symbol(self):
        return self._symbol

    def apply(self, left: bool, right: bool=False):
        if self.isUnary:
            return self._fn(left)
        return self._fn(left, right)
    
    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return """[Operator] symbol: '{0}'""".format(self.symbol)
    
    def __eq__(self, val:str):
        if isinstance(val, str):
            return self.symbol == val
        if isinstance(val, Operator):
            return self.symbol == val.symbol
        raise Exception("[CompareError] Cannot compare object {0} with object{1}".format(type(val), type(Operator)))

# Rule rmin go first
OPERATORS = (
    # symbol: str, name: str, isBinary: bool, fn_operation: Callable[[int, int], int], priority: int, isUnary
    ('or',   'OR',     lambda p, q: p or q, 2, False),
    ('and',  'AND',    lambda p, q: p and q, 2, False),
    ('not',  'NOT',    lambda q: not q, 1, True),
    ('imply','IMPLY',  lambda p, q: (not p) or q, 3, False),  
    ('equal','EQU',    lambda p, q: p == q, 3, False),
    )
OPERATORS = list(map(lambda x: Operator(*x), OPERATORS))


class Parser:
    def __init__(self, s: str, *, operators: List[Operator] = OPERATORS):
        # clean the Token
        Token.removeAll()

        # init variables
        self.s: str = s
        self.operators = operators
        self._tokenCount: int = 0
        self.postfixed: List[Operator|Token] = self.__postfix(self.__operation_to_list(s))
        
    @property
    def tokenCount(self) -> int:
        return self._tokenCount

    def __operation_to_list(self, s: str)-> List[str]:
        operators = tuple([operator.symbol for operator in self.operators])
        _s = re.split(r'(\(|\)|%s)'%('|'.join(operators)), s)
        # remove whitespace in splited expression
        _s = map(lambda val: val.replace(' ', ''), _s)
        # remove blanck elements
        _s = filter(lambda token: len(token) > 0, _s)
        return list(_s)

    def __postfix(self, s: str) -> List[Operator|Token]:
        """Transform the expression to its Postfixed form

        Returns:
            [Operator|Token]: postfixed expression
        """
        stack = list() # LIFO store operators
        operators = {
            operator.symbol : operator for operator in self.operators
        }
        OPENPARENTHESIS = '('
        CLOSEPARENTHESIS = ')'
        operators_symbols: List[str] = operators.keys()
        operations: List[Token|Operator] = list()
        s: List[str] = self.__operation_to_list(self.s)
        while len(stack) > 0 or len(s) > 0:
            # Left To Right (LR)
            if len(s) > 0:
                # Get the current element from input(Token | Operator | parenthesis)
                val: str = s.pop(0)        
            else:
                # No more tokens to process in input
                # Add remaining operations(in the stack) to the main queue
                for _ in range(len(stack)):
                    operations.append(stack.pop())
                continue
            # Check if current element is an operator
            if val in operators_symbols:
                # is the stack (containing operator) empty?
                val: Operator = operators.get(val)
                if len(stack) > 0:
                    # the stack is not empty
                    # get the top of the stack
                    top: Operator = stack.pop()
                    # Is the topest element a openparenthesis?
                    if top != OPENPARENTHESIS:
                        # The top element in operator_stack is not a parenthesis
                        # priority is DESC (rmin has higher priority)
                        if top.priority <= val.priority:
                            # add higher priority element into the queue
                            operations.append(top)
                            stack.append(val)
                            continue
                        else:
                            stack.append(top)
                            stack.append(val)
                            continue
                    else:
                        # Is parenthesis
                        stack.append(top)
                        stack.append(val)
                        continue
                else:
                    # The stack is empty
                    stack.append(val)
                    continue
                
            elif val == OPENPARENTHESIS:
                stack.append(val)
                continue
            elif val == CLOSEPARENTHESIS:
                # Remove all element from the stack until OPENPARENTHESIS
                while len(stack) > 0 :
                    top = stack.pop()
                    if top == OPENPARENTHESIS:
                        break
                    else:
                        operations.append(top)
                continue
            else:
                # the current element is a variable | constant
                val: Token = Token(val)
                operations.append(val)
        self._tokenCount = Token.getInstancesCount()
        return operations

    def evalExpression(self, values: List[bool])->bool:
        """Evaluate the expression

        Args:
            values (List[bool]): vector having the dimension of the number of variables in the expression

        Returns:
            bool: result of the expression
        """
        assert len(values) == Token.getInstancesCount(), '[NotEnoughValuesError] numbers of variables in expression: {0} number of variable on input: {1}'.format(Token.getInstancesCount, len(values))
        # Assigning value to variables using @arg: values
        for i, variable in enumerate(Token.getTokens()):
            variable.value = values[i]

        expression = [val for val in self.postfixed]  # copy the current postfix expresion
        stack: List[bool] = list()  # FIFO
        
        while len(expression) > 0:
            val = expression.pop(0)
            # Check if its a number
            if isinstance(val, Token):
                # Add the number to the stack
                stack.append(val.value)
            # If the current value is not a constant or a variable
            # it must be an operator
            else:
                # Three address code 
                try:   
                    operator: Operator = val
                    if operator.isUnary:
                        # Is The stack empty ?
                        if len(stack) == 0:
                            temp_stack = list()
                            temp_stack.append('(')  # '(' is just a marker
                            _val = expression.pop(0)
                            while not isinstance(_val, Token):  # POP all unary operators till getting a Token
                                temp_stack.append(_val)
                                _val = expression.pop()
                            _val = _val.value
                            _ = temp_stack.pop(-1)
                            while _ != '(':
                                _val = _.apply(_val)
                                _ = temp_stack.pop(-1)
                            stack.append(operator.apply(_val))
                        else:
                            left = stack.pop(-1)
                            stack.append(operator.apply(left))
                        continue
                    # it must be an operator here if the poped elements(left=operator,right=operator) is an operator(like in ----1-----3)
                    # for example
                    # what are the operand of the current operator
                    right: Token | None = None
                    left: Token | None = None
                    right = stack.pop(-1)
                    left  = stack.pop(-1)
                    stack.append(operator.apply(left, right))            
                except IndexError: # pop an empty stack
                    right: bool = False if right is None else right
                    left: bool = False if left is None else left
                    stack.append(operator.apply(left, right))
                    continue
                except Exception as e:
                    print('[compileStringError] unknown operator: {0} Error: {1}'.format(val, e))
                    raise Exception('[OperationAborted]')
                    
        return stack.pop()
    
    def bruteforce(self, verbose: bool=False) -> List[dict]:
        """Bruteforce the expression to find the solution

        Args:
            verbose (bool, optional): enable logs printing. Defaults to False.

        Returns:
            List[dict]: solutions of the expression e.g. [{x1: False, x2: True, ...}, {x1: True, x2: True}, ...]
        """
        values = list(product([True, False], repeat=self.tokenCount))
        solutions = []
        for value in values:
            if verbose:
                print('Expression: ', self.s)
            # print('Postfixed: ', self.postfixed)
            ret = self.evalExpression(list(value))
            if ret:
                solutions.append(value)
            if verbose: 
                for i, token in enumerate(Token.getTokens()):
                    print('%s : %s; ' % (token, value[i]), end='')
                print(end='\n')
                print("ret: ", ret)
                print('-'*10)
        if not len(solutions) > 0:
            return solutions
        variable_keys = [variable.name for variable in Token.getTokens()]
        solutions = [{variable_keys[i]: value for i, value in enumerate(solution)} for solution in solutions]
        return solutions
