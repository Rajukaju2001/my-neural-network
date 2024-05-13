import math


class MyNet:
    """
    A class for building and training computational graphs.

    Attributes:
        data (float): The value of the node.
        grad (float): The gradient of the node.
        _prev (set of MyNet): The set of parent nodes.
        _op (str): The operation that was applied to create the node.
        _backward (function): The function to compute the gradient of the node.
    """

    def __init__(self, data, _children=(), _op="") -> None:
        """
        Initialize a new node with the given data and parent nodes.

        Args:
            data (float): The value of the node.
            _children (set of MyNet, optional): The set of parent nodes. Defaults to ().
            _op (str, optional): The operation that was applied to create the node. Defaults to "".
        """
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self) -> str:
        """
        Return a string representation of the node.

        Returns:
            str: A string representation of the node.
        """
        return f"MyNet({self.data} | {self.grad})"

    def __add__(self, other):
        """
        Add two nodes together.

        Args:
            other (MyNet or float): The node or value to add.

        Returns:
            MyNet: A new node representing the sum of the two nodes.
        """
        other = other if isinstance(other, MyNet) else MyNet(other)
        out = MyNet(self.data + other.data, (self, other), "+")

        def backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = backward
        return out

    def __radd__(self, other):
        """
        Add a value to a node.

        Args:
            other (float): The value to add.

        Returns:
            MyNet: A new node representing the sum of the node and the value.
        """
        return self + other

    def __mul__(self, other):
        """
        Multiply two nodes together.

        Args:
            other (MyNet or float): The node or value to multiply.

        Returns:
            MyNet: A new node representing the product of the two nodes.
        """
        other = other if isinstance(other, MyNet) else MyNet(other)
        out = MyNet(self.data * other.data, (self, other), "*")

        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = backward
        return out

    def __rmul__(self, other):
        """
        Multiply a value by a node.

        Args:
            other (float): The value to multiply.

        Returns:
            MyNet: A new node representing the product of the node and the value.
        """
        return self * other

    def tanh(self):
        """
        Apply the hyperbolic tangent function to the node.

        Returns:
            MyNet: A new node representing the result of the hyperbolic tangent function.
        """
        out = MyNet(math.tanh(self.data), (self,), "tanh")

        def backward():
            self.grad += (1 - out.data**2) * out.grad

        out._backward = backward
        return out

    def backward(self):
        """
        Compute the gradients of all nodes in the computational graph using backpropagation.
        """

        def topological_sort(node):
            """
            Perform a depth-first search to compute the topological order of the nodes.
            """
            visited = set()
            order = []

            def dfs(node):
                visited.add(node)
                for child in node._prev:
                    if child not in visited:
                        dfs(child)
                order.append(node)

            dfs(node)
            return order[::-1]  # Reverse the order to get topological sort

        order = topological_sort(self)
        for node in order:
            node._backward()
