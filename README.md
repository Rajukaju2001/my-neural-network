# MyNet

MyNet is a Python library for building and training computational graphs. It allows you to create nodes with values and gradients, and perform operations such as addition, multiplication, and hyperbolic tangent on them. The library also supports backpropagation for computing gradients of all nodes in the computational graph.

## Installation

To install MyNet, you can clone this repository and run the following command:

```
pip install .
```

## Usage

To create a new node with a value and gradients, you can use the `MyNet` class:

```python
node = MyNet(data=1.0)
print(node)  # MyNet(1.0 | 0.0)
```

You can perform operations on nodes using the `+`, `*`, and `tanh` operators:

```python
node1 = MyNet(data=1.0)
node2 = MyNet(data=2.0)
node3 = node1 + node2
node4 = node1 * node2
node5 = node1.tanh()

print(node3)  # MyNet(3.0 | 0.0)
print(node4)  # MyNet(2.0 | 0.0)
print(node5)  # MyNet(0.7615941559557649 | 0.0)
```

To compute gradients of all nodes in the computational graph, you can use the `backward` method:

```python
node1 = MyNet(data=1.0)
node2 = MyNet(data=2.0)
node3 = node1 + node2
node4 = node3 * node3
node5 = node4.tanh()

node5.backward()

print(node1.grad)  # 4.0
print(node2.grad)  # 4.0
print(node3.grad)  # 8.0
print(node4.grad)  # 4.0
print(node5.grad)  # 1.0
```

## Contributing

If you would like to contribute to MyNet, please open a pull request with your proposed changes.

## License

MyNet is released under the MIT License. See the LICENSE file for more information.
