def recursive(f):
  '''A decorator to wrap a method of a HierarchicalModel to (1) firstly invoke the method,
     and (2) call all the direct descendants of the HierarchicalModel to also invoke the method.
     if the descdendant also decorate the method with the 'recursive', this process will be replicated
     until it reaches object with no descendant or the method is not marked with recursive.
  '''
  # Wrapper to the method to call recursively, instead of invoking only f()
  def rec_f(obj, *args, **kwargs):
    assert(issubclass(obj.__class__, HierarchicalModel))
    # Reflect the method name
    name = f.__name__
    # Invoke the method manually with its arguments
    f(obj, *args, **kwargs)
    # For all the descendant, also invoke the method with same argument
    for member in obj.get_hier_children():
      if hasattr(member, name):
        getattr(member, name)(*args, **kwargs)

  # Return the wrapper
  return rec_f

def recursive_assign(f):
  ''' A decorator that behaves the same way as recursive but keep returning the context of a previous
      method invocation and pass the context to the next. We assume that the decorated method will use / modify
      the context as needed.
  '''
  def rec_f(obj, *args, **kwargs):
    assert(issubclass(obj.__class__, HierarchicalModel))
    name = f.__name__
    kwargs["context"] = f(obj, *args, **kwargs)
    for member in obj.get_hier_children():
      if hasattr(member, name):
        kwargs["context"] = getattr(member, name)(*args, **kwargs)
    return kwargs["context"]

  return rec_f

def recursive_sum(f):
  ''' A decorator that behaves the same way as recursive but summing up all the non None results.
  '''
  def rec_f(obj, *args, **kwargs):
    assert(issubclass(obj.__class__, HierarchicalModel))
    name = f.__name__
    result_parent = f(obj, *args, **kwargs)
    for member in obj.get_hier_children():
      if hasattr(member, name):
        result_child = getattr(member, name)(*args, **kwargs)
        if result_child is not None:
          if result_parent is None:
            result_parent = result_child
          else:
            result_parent += result_child
    return result_parent

  return rec_f


class HierarchicalModel(object):
  ''' Hierarchical Model interface '''

  def register_hier_child(self, child):
    if hasattr(child, "register_hier_child"):
      if not hasattr(self, "_hier_children"):
        self._hier_children = []
      self._hier_children.append(child)
    else:
      print("Skipping hierarchical construction:", child.__class__.__name__,
            "is not a subclass of HierarchicalModel")
  def get_hier_children(self):
    if hasattr(self, "_hier_children"): return self._hier_children
    else: return []

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

  @recursive
  def initialize_generator(self, **kwargs):
    pass

  @recursive
  def new_epoch(self):
    pass

  @recursive
  def set_train(self, val):
    pass

  @recursive
  def start_sent(self):
    pass

  def calc_loss(self, src, trg, src_mask=None, trg_mask=None):
    raise NotImplementedError()

  @recursive_sum
  def calc_additional_loss(self, reward):
    ''' Calculate reinforce loss based on the reward
    :param reward: The default is log likelihood (-1 * calc_loss).
    '''
    return None

