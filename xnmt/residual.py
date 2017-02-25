from dynet import *

class ResidualRNNBuilder:
    """
    Builder for RNNs that implements additional residual connections between layers.
    """

    def __init__(self, num_layers, input_dim, hidden_dim, model, rnn_builder_factory):
        """
        @param num_layers: depth of the RNN (> 0)
        @param input_dim: size of the inputs
        @param hidden_dim: size of the outputs (and intermediate layer representations)
        @param model
        @param rnn_builder_factory: RNNBuilder subclass, e.g. LSTMBuilder
        """
        assert num_layers > 0
        self.builder_layers = []
        self.builder_layers.append(rnn_builder_factory(1, input_dim, hidden_dim, model))
        for _ in range(num_layers - 1):
            self.builder_layers.append(rnn_builder_factory(1, hidden_dim, hidden_dim, model))

    def whoami(self):
        return "ResidualRNNBuilder"

    def set_dropout(self, p):
        for l in self.builder_layers:
            l.set_dropout(p)

    def disable_dropout(self):
        for l in self.builder_layers:
            l.disable_dropout()

    def add_inputs(self, es):
        """
        returns the list of state pairs (stateF, stateB) obtained by adding
        inputs to both forward (stateF) and backward (stateB) RNNs.

        @param es: a list of Expression

        see also transduce(xs)

        .transduce(xs) is different from .add_inputs(xs) in the following way:

            .add_inputs(xs) returns a list of RNNState pairs. RNNState objects can be
             queried in various ways. In particular, they allow access to the previous
             state, as well as to the state-vectors (h() and s() )

            .transduce(xs) returns a list of Expression. These are just the output
             expressions. For many cases, this suffices.
             transduce is much more memory efficient than add_inputs.
        """
        es = self.builder_layers[0].initial_state().transduce(es)
        for l in self.builder_layers[1:-1]:
            es += l.initial_state().transduce(es)
        return self.builder_layers[-1].initial_state().add_inputs(es) + es

    def transduce(self, es):
        """
        returns the list of output Expressions obtained by adding the given inputs
        to the current state, one by one, to both the forward and backward RNNs,
        and concatenating.

        @param es: a list of Expression

        see also add_inputs(xs), including for explanation of differences between
        add_inputs and this function.
        """
        es = self.builder_layers[0].initial_state().transduce(es)
        for l in self.builder_layers[1:]:
            es += l.initial_state().transduce(es)
        return es
