from neuralpp.symbolic.expression import Expression, FunctionApplication, Variable, Constant
from abc import ABC


class BasicExpression(Expression, ABC):
    pass


class BasicVariable(BasicExpression, Variable):
    pass


class BasicConstant(BasicExpression, Constant):
    pass


class BasicFunctionApplication(BasicExpression, FunctionApplication):
    pass
