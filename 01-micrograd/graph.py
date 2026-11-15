import os
from graphviz import Digraph

_counter = 0  # numbers the SVGs so each draw_dot call gets its own file
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  

def trace(root):
    # builds a set of all nodes and edges in a graph
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


# draws the computation graph of a Value (its values, ops and how they connect)
def draw_dot(root, name=None, folder=os.path.join(ROOT_DIR, 'graphs')):
    global _counter
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})  # LR = left to right

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node for it
        dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
        if n._op:
            # if this value is a result of some operation, create an op node for it
            dot.node(name = uid + n._op, label = n._op)
            # and connect this node to it
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    # save as SVG in folder (<project>/graphs by default) without opening a viewer
    _counter += 1
    if name is None:
        name = f"graph_{_counter:03d}"
    os.makedirs(folder, exist_ok=True)
    path = dot.render(os.path.join(folder, name), cleanup=True)
    print(f"saved {path}")

    return dot