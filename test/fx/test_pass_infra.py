# Owner(s): ["oncall: fx"]

import torch
import torch.fx as fx

from torch.testing._internal.common_utils import TestCase
from torch.fx.passes.infra.pass_base import PassResult
from torch.fx.passes.infra.pass_manager import (
    PassManager,
    this_before_that_pass_constraint,
    topological_sort_passes,
    post_pass_hook,
)

def replace_add_with_mul_pass(gm):
    modified = False
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target == torch.add:
            node.target = torch.mul
            modified = True
    return PassResult(gm, modified)

def replace_mul_with_div_pass(gm):
    modified = False
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target == torch.mul:
            node.target = torch.div
            modified = True
    return PassResult(gm, modified)

class AddModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = torch.add(x, x)
        z = torch.add(y, x)
        return z


class TestPassManager(TestCase):
    def test_pass_manager(self):
        """
        Tests that the pass manager runs the passes correctly.
        """

        m = AddModule()
        traced_m = torch.fx.symbolic_trace(m)
        pm = PassManager(passes=[replace_add_with_mul_pass, replace_mul_with_div_pass], steps=5)

        pm.validate_constraints()
        self.assertEqual(len(pm.passes), 2)

        res = pm(traced_m)
        modified_m = res.graph_module
        assert isinstance(modified_m, fx.GraphModule)

        # Check that all call_function nodes are divs
        for node in modified_m.graph.nodes:
            if node.op == "call_function":
                self.assertEqual(node.target, torch.div)

    def test_this_before_that_pass_constraint(self):
        """
        Tests the construction of constraints
        """
        passes = [lambda x: 2 * x for _ in range(10)]
        pm = PassManager(passes)

        # add unfulfillable constraint
        pm.add_constraint(this_before_that_pass_constraint(passes[-1], passes[0]))

        with self.assertRaises(RuntimeError):
            pm.validate_constraints()


    def test_pass_manager_checks(self):
        """
        Tests that users can add in check functions correctly
        """
        m = AddModule()
        traced_m = fx.symbolic_trace(m)
        pm = PassManager(passes=[replace_add_with_mul_pass, replace_mul_with_div_pass])

        def check_div_target(graph_module):
            for node in graph_module.graph.nodes:
                if node.op == "call_function" and node.target != torch.div:
                    raise ValueError("Target should be div!")
        pm.add_checks(check_div_target)

        with self.assertRaises(ValueError):
            pm(traced_m)

    def test_pass_manager_bad_checks(self):
        """
        Checks that we error if we pass in a check function with the wrong parameters
        """
        def check_bad_args(graph_module, i):
            pass

        pm = PassManager()
        self.assertRaises(TypeError, pm.add_checks, check_bad_args)

    def test_pass_manager_hooks(self):
        """
        Tests that hooks are added correctly
        """
        num_call_func = 0

        def count_num_call_func(graph_module):
            nonlocal num_call_func
            for node in graph_module.graph.nodes:
                if node.op == "call_function":
                    num_call_func += 1

        m = AddModule()
        traced_m = fx.symbolic_trace(m)
        PassManager(passes=[
            post_pass_hook(replace_add_with_mul_pass, *(count_num_call_func,)),
            post_pass_hook(replace_mul_with_div_pass, *(count_num_call_func,)),
        ])(traced_m)

        self.assertEqual(num_call_func, 4)


    def test_topological_sort(self):
        """
        Tests that passes are correctly ordered based on contraints.
        """

        def pass0(x):
            return x

        def pass1(x):
            return x + 1

        def pass2(x):
            return x + 2

        def pass3(x):
            return x + 3

        def pass4(x):
            return x + 4

        def pass5(x):
            return x + 5

        # Not passing any constraints should keep the original order
        passes = [pass0, pass1, pass2, pass3, pass4, pass5]
        sorted = topological_sort_passes(passes, [])
        self.assertEqual(sorted, (passes, False))

        # Graph that we are constructing:
        #     5 -> 0 <- 4
        #     |         |
        #     2 -> 3 -> 1
        # Which has a possible topological order of: [4, 5, 0, 2, 3, 1]
        passes = [pass0, pass1, pass2, pass3, pass4, pass5]
        constraints = [
            this_before_that_pass_constraint(passes[5], passes[0]),
            this_before_that_pass_constraint(passes[5], passes[2]),
            this_before_that_pass_constraint(passes[4], passes[0]),
            this_before_that_pass_constraint(passes[4], passes[1]),
            this_before_that_pass_constraint(passes[2], passes[3]),
            this_before_that_pass_constraint(passes[3], passes[1]),
        ]
        sorted = topological_sort_passes(passes, constraints)
        self.assertEqual(sorted, ([pass4, pass5, pass2, pass3, pass1, pass0], False))

        # Circular dependency should result in the circular_dep flag being set
        passes = [pass0, pass1, pass2]
        constraints = [
            this_before_that_pass_constraint(passes[0], passes[1]),
            this_before_that_pass_constraint(passes[1], passes[2]),
            this_before_that_pass_constraint(passes[2], passes[0]),
        ]
        sorted = topological_sort_passes(passes, constraints)
        self.assertEqual(sorted, ([pass2, pass0, pass1], True))

    def test_validation_inputs(self):
        """
        Tests that the validation function runs correctly.
        """
        m = AddModule()
        traced_m = torch.fx.symbolic_trace(m)
        pm = PassManager(passes=[replace_add_with_mul_pass, replace_mul_with_div_pass], run_checks_after_each_pass=True)

        with self.assertRaises(RuntimeError):
            pm(traced_m, **{"input": torch.ones(1, 3), "rtol": 1e-05})