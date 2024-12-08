from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple, Protocol


# ## Task 1.1

# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    # Convert the input values to a list
    vals_list = list(vals).copy()

    # Compute f(x + epsilon)
    vals_list_plus = vals_list.copy()
    vals_list_plus[arg] += epsilon
    f_plus = f(*vals_list_plus)

    # Compute f(x - epsilon)
    vals_list_minus = vals_list.copy()
    vals_list_minus[arg] -= epsilon
    f_minus = f(*vals_list_minus)

    # Compute the central difference
    return (f_plus - f_minus) / (2.0 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulates the derivative for this variable.

        This variable should only be called during autodifferentiation on leaf variables.

        Args:
        ----
            x: The value to be added to the current derivative.

        """
        ...

    @property
    def unique_id(self) -> int:
        """Returns the unique ID of this variable."""
        ...

    def is_leaf(self) -> bool:
        """Returns True if this variable is a leaf."""
        ...

    def is_constant(self) -> bool:
        """Returns True if this variable is constant."""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Returns the parent variables of this variable."""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Computes the chain rule for this variable.

        Args:
        ----
            d_output: The derivative of the output with respect to this variable.

        Returns:
        -------
            An iterable of tuples, where each tuple contains a parent variable and its derivative.

        """
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    # ASSIGNMENT 1.4

    # Initialize empty list to store variables in topological order
    order: List[Variable] = []
    # Set to keep track of visited variables
    seen = set()

    def visit(var: Variable) -> None:
        # Skip if variable was already visited or is constant
        if var.unique_id in seen or var.is_constant():
            return
        # For non-leaf variables, recursively visit their parents first
        if not var.is_leaf():
            for parent in var.parents:
                # Only visit non-constant parents
                if not parent.is_constant():
                    visit(parent)
        # Mark variable as seen
        seen.add(var.unique_id)
        # Add variable to start of order list (reverse topological order)
        order.insert(0, var)

    # Start DFS from the rightmost variable
    visit(variable)
    # Return variables in topological order
    return order

    # END ASSIGNMENT 1.4


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order tocompute derivatives for the leave nodes.

    Args:
    ----
        variable: The right-most variable
        deriv: Its derivative that we want to propagate backward to the leaves.

    """
    # Call topological sort
    sorted_variables = topological_sort(variable)

    # Create dict of Variables' unique IDs and their derivatives
    derivatives = {}

    # Add the derivative of the input variable first
    derivatives[variable.unique_id] = deriv

    # For each node in backward order:
    for var in sorted_variables:
        # Get the derivative of the current variable (always exists according to the topological sort algorithm)
        deriv = derivatives[var.unique_id]

        # If the variable is a leaf, add its final derivative using the accumulate_derivative method
        if var.is_leaf():
            var.accumulate_derivative(deriv)

        # If the variable is not a leaf
        else:
            # Call the chain_rule method to compute the derivatives of the parents
            for parent, parent_deriv in var.chain_rule(deriv):
                # Skip constant variables since their derivatives don't need to be computed
                if parent.is_constant():
                    continue
                # Initialize derivative for parent to 0 if not already in derivatives dict
                derivatives.setdefault(parent.unique_id, 0)
                # Add the computed parent derivative to any existing derivative
                # This handles the case where a variable has multiple children
                derivatives[parent.unique_id] += parent_deriv


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returns the saved values from the forward method."""
        return self.saved_values
