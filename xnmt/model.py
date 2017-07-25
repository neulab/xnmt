import functools

def recursive(f):
  def rec_f(obj, *args, **kwargs):
    name = f.__name__
    # Invoke the method
    f(obj, *args, **kwargs)
    # Invoker the method of all children, if the children has that method
    for member in obj._hier_children:
      if hasattr(member, name):
        getattr(member, name)(*args, **kwargs)

  return rec_f

class HierarchicalModel(object):
  ''' Hierarchical Model interface '''
  def __init__(self):
    self._hier_children = []

  def register_hier_child(self, child):
    if hasattr(child, "register_hier_child"):
      self._hier_children.append(child)
    else:
      print("Skipping hierarchical construction:", child.__class__.__name__,
            "is not a subclass of HierarchicalModel")

class GeneratorModel(HierarchicalModel):
  def generate_output(self, *args, **kwargs):
    # Generate the output
    generation_output = self.generate(*args, **kwargs)
    # Post process it
    if hasattr(self, "post_processor"):
      generation_output = self.post_processor.process_outputs(generation_output)[0]
    return generation_output

  def generate(self, *args, **kwargs):
    raise NotImplementedError()

  def initialize(self, system_args):
    pass

  def set_vocabs(self, src_vocab, trg_vocab):
    self.src_vocab = src_vocab
    self.trg_vocab = trg_vocab

  def set_post_processor(self, post_processor):
    self.post_processor = post_processor

