import model

def recursive(f):
  def rec_f(obj, *args, **kwargs):
    assert(issubclass(obj.__class__, model.HierarchicalModel))
    name = f.__name__
    # Invoke the method
    f(obj, *args, **kwargs)
    for member in obj._hier_children:
      if hasattr(member, name):
        getattr(member, name)(*args, **kwargs)

  return rec_f

def recursive_assign(f):
  def rec_f(obj):
    assert(issubclass(obj.__class__, model.HierarchicalModel))
    name = f.__name__
    ctxt = f(obj, None)
    for member in obj._hier_children:
      if hasattr(member, name):
        ctxt = getattr(member, name)(ctxt)
    return ctxt

  return rec_f

