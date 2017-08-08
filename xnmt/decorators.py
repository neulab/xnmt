import model

def recursive(f):
  '''A decorator to wrap a method of a HierarchicalModel to (1) firstly invoke the method,
     and (2) call all the direct descendants of the HierarchicalModel to also invoke the method.
     if the descdendant also decorate the method with the 'recursive', this process will be replicated
     until it reaches object with no descendant or the method is not marked with recursive.
  '''
  # Wrapper to the method to call recursively, instead of invoking only f()
  def rec_f(obj, *args, **kwargs):
    assert(issubclass(obj.__class__, model.HierarchicalModel))
    # Reflect the method name
    name = f.__name__
    # Invoke the method manually with its arguments
    f(obj, *args, **kwargs)
    # For all the descendant, also invoke the method with same argument
    for member in obj._hier_children:
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
    assert(issubclass(obj.__class__, model.HierarchicalModel))
    name = f.__name__
    kwargs["context"] = f(obj, *args, **kwargs)
    for member in obj._hier_children:
      if hasattr(member, name):
        kwargs["context"] = getattr(member, name)(*args, **kwargs)
    return kwargs["context"]

  return rec_f

def recursive_sum(f):
  ''' A decorator that behaves the same way as recursive but summing up all the non None results.
  '''
  def rec_f(obj, *args, **kwargs):
    assert(issubclass(obj.__class__, model.HierarchicalModel))
    name = f.__name__
    result_parent = f(obj, *args, **kwargs)
    for member in obj._hier_children:
      if hasattr(member, name):
        result_child = getattr(member, name)(*args, **kwargs)
        if result_child:
          if not result_parent:
            result_parent = result_child
          else:
            result_parent += result_child
    return result_parent

  return rec_f

