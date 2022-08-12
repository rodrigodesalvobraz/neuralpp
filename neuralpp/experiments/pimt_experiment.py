from neuralpp.symbolic.basic_expression import BasicExpression
from neuralpp.symbolic.expression import Variable
from neuralpp.symbolic.polynomial_approximation import get_normal_piecewise_polynomial_approximation

x = Variable("x")
mu1 = Variable("mu1")
mu2 = Variable("mu2")

# P(x, mu1, mu2) = Normal(x | mu1, 1.0) * Normal(mu1 | mu2, 1.0) * Normal(mu2 | 0.0, 1.0) propto
joint = get_normal_piecewise_polynomial_approximation(x, mu1, 1.0) \
        * \
        get_normal_piecewise_polynomial_approximation(mu1, mu2, 1.0) \
        * \
        get_normal_piecewise_polynomial_approximation(mu2, BasicExpression.new_constant(0.0), 1.0)

# P(x, mu2) propto
# phi_x_mu2 = sum_mu1 joint   # we use phi to indicate an unnormalized distribution

# P(x) propto
# phi_x = sum_mu1 phi_x_mu2

# P(mu2 | x) = phi_x_mu2 / phi_x

# This is symbolic in mu2 and x. To obtain the posterior for mu2 for a given evidence (observation) x,
# simply plug it and evaluate.


