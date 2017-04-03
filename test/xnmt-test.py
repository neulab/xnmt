import sys

sys.argv.append("--dynet-viz")
sys.path.append("../xnmt")

import dynet as dy
import residual

m = dy.Model()
v = residual.ResidualBiRNNBuilder(4, 4, 2, m, dy.SimpleRNNBuilder)

data1 = dy.vecInput(4)
data1.set([1, 2, 3, 4])
data2 = dy.vecInput(4)
data2.set([5, 6, 7, 8])
v.transduce([data1, data2])

dy.print_graphviz()
