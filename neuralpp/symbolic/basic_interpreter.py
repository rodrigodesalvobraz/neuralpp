from neuralpp.symbolic.interpreter import Interpreter
from neuralpp.symbolic.expression import Expression, FunctionApplication, Constant, Variable
from neuralpp.symbolic.basic_expression import BasicConstant


class BasicInterpreter(Interpreter):
    def eval(self, expression: Expression, context: Expression):
        """
        In this BasicInterpreter, `context` is ignored and assumed to be True.
        `expression` is assumed to only (recursively) contain FunctionaApplication and Constant.
        The function raises Error if it
        - encounters a Variable, standalone or in arguments;
        - encounters a Variable as the function (it's an "uninterpreted function")
        - the function raises Error when called on the arguments. (this is implicitly raised)
        Note the function does not check `context`, since if the result can be evaluated under True context,
        it must also can be evaluated under a weaker one.
         """
        # pattern matching in Python: https://peps.python.org/pep-0636/
        match expression:
            # there's three cases of `function` that BasicInterpreter can eval():
            # 1. a Python callable, which we'll directly call
            # 2. an uninterpreted function, which is not callable. We raise Error when encounter it.
            case FunctionApplication(function=Constant(value=python_callable), arguments=args):
                # * is used to turn a list into "args": https://docs.python.org/2/reference/expressions.html#calls
                return python_callable(*[self.eval(e, BasicConstant(True)) for e in args])
            case FunctionApplication(function=Variable(name=f), arguments=args):
                raise AttributeError(f"Function {f} is uninterpreted. It cannot be evaluated by BasicInterpreter.")
            case Constant(value=value):
                return value
            case Variable(name=_):
                raise AttributeError("BasicInterpreter expects no variables when it eval()")
