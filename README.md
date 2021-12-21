# SATBruteforce
Solve a boolean expression using bruteforce
# Operators
 - EQUA     ( == )
 - IMPLY    ( => )
 - OR       ( || )
 - AND      ( && )
 - NOT      ( ~  )
 # example : 
 - expression : '(p) AND (p IMPLY q) AND (q IMPLY r)'
 - parser = Parser(expression)
 - solutions = parser.bruteforce(verbose=False)
 - return : {'p': True, 'q': True, 'r': True}