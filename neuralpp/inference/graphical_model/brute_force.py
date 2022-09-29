from neuralpp.util.group import Group


class BruteForce:
    @staticmethod
    def run(query, factors):
        total_product = Group.product(factors)
        other_variables = {v for f in factors for v in f.variables if v != query}
        marginal = (total_product ^ other_variables).normalize()
        return marginal
