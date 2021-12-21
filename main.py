# -*- coding: utf-8 -*-
"""
@author : Princy Rasolonjatovo
@email  : princy.m.rasolonjatovo@gmail.com
@github : princy-rasolonjatovo
"""
from solver import Parser


if __name__ == '__main__':
    # TESTS
    expressions = ['(p) AND (p IMPLY q) AND (q IMPLY r)',
        'a AND NOT b',
        '(a OR b) AND (a OR NOT b)']

    expression = expressions[0]
    parser = Parser(expression)
    solutions = parser.bruteforce(verbose=False)
    # printing solutions
    print('expression: ', expression)
    if len(solutions) > 0:
        print('solution%s : '%('s' if len(solutions) > 1 else ''))
        for solution in solutions:
            print('params: ', solution)
    else:
        print('no solutions')