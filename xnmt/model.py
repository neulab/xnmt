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


