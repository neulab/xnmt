"""
This module implements a global event mechanism in XNMT.

Events are handled globally, i.e. caller and handler do not need to know about each other, and it is not possible to
limit the scope of an event. Events are very useful, but should be used sparingly and only when triggering global
events is not expected to hurt modularity. Event handling involves two parts:

- Registering an event caller. Callers are always module-level functions, and registration simply means decorating them
  as such:

  @register_xnmt_event
  def my_event():
    pass

  The ``xnmt.event_triggers`` module is the general place to add such callers.

- Defining event handlers. Event handlers are always *class methods*. The following
  must hold:
  - handlers are named 'on_' + name of the event, e.g. on_my_event
  - method arguments must be consistent
  - handlers must be decorated with @handle_xnmt_event
  - the handler's __init__ method must be decorated with @register_xnmt_handler(self)

  example:

  class AnotherObject(object):
    @register_xnmt_handler
    def __init__(self):
      ...
    @handle_xnmt_event
    def on_my_event():
      # do something


  Events can also return values. To make use of return values, 2 special decorators are available:
  - @register_xnmt_event_assign assumes a special keyword argument named "context".
    The return value is an updated value for this context argument, and will then be passed on to the next event handler
  - @register_xnmt_event_sum here the return values of all handlers are summed up and returned.
"""

from functools import wraps

from xnmt import logger

def clear():
  """
  clear handler (mostly useful to remove side effects when running a series of unit tests)
  """
  global handler_instances, handler_method_names, event_names
  handler_instances = []
  handler_method_names = set()

handler_instances = []
handler_method_names = set()
event_names = set()

def register_xnmt_handler(f):
  @wraps(f)
  def wrapper(obj, *args, **kwargs):
    f(obj, *args, **kwargs)
    handler_instances.append(obj)
  return wrapper

def register_xnmt_event(f):
  @wraps(f)
  def wrapper(*args, **kwargs):
    assert handler_method_names <= event_names, "detected handler for non-existant event: {}".format(
      handler_method_names - event_names)
    f(*args, **kwargs)
    for handler in handler_instances:
      bound_handler = getattr(handler, "on_" + f.__name__, None)
      if bound_handler:
        ret = bound_handler(*args, **kwargs)
        if type(ret)!=tuple or len(ret)!=2 or ret[1][3:]!=f.__name__:
          raise RuntimeError("attempted to call unregistered handler {}".format(f.__name__))
  event_names.add(f.__name__)
  return wrapper

def register_xnmt_event_assign(f):
  @wraps(f)
  def wrapper(*args, **kwargs):
    assert "context" in kwargs, "register_xnmt_event_assign requires \"context\" in kwargs"
    assert handler_method_names <= event_names, "detected handler for non-existant event: {}".format(
      handler_method_names - event_names)
    kwargs["context"] = f(*args, **kwargs)
    for handler in handler_instances:
      bound_handler = getattr(handler, "on_" + f.__name__, None)
      if bound_handler:
        ret = bound_handler(*args, **kwargs)
        if type(ret)!=tuple or len(ret)!=2 or ret[1][3:]!=f.__name__:
          raise RuntimeError("attempted to call unregistered handler {}".format(f.__name__))
        kwargs["context"] = ret[0]
    return kwargs["context"]
  event_names.add(f.__name__)
  return wrapper

def register_xnmt_event_sum(f):
  @wraps(f)
  def wrapper(*args, **kwargs):
    assert handler_method_names <= event_names, "detected handler for non-existant event: {}".format(
      handler_method_names - event_names)
    res = f(*args, **kwargs)
    for handler in handler_instances:
      bound_handler = getattr(handler, "on_" + f.__name__, None)
      if bound_handler:
        ret = bound_handler(*args, **kwargs)
        if type(ret)!=tuple or len(ret)!=2 or ret[1][3:]!=f.__name__:
          raise RuntimeError("attempted to call unregistered handler {}".format(f.__name__))
        tmp = ret[0]
        if res is None: res = tmp
        elif tmp is not None: res = res + tmp
    return res
  event_names.add(f.__name__)
  return wrapper

def handle_xnmt_event(f):
  @wraps(f)
  def wrapper(obj, *args, **kwargs):
    try:
      return f(obj, *args, **kwargs), f.__name__
    except:
      logger.error("Error handling xnmt event at object:", obj.__class__.__name__)
      raise
  assert f.__name__.startswith("on_"), "xnmt event handlers must be named on_*, found {}".format(f.__name__)
  handler_method_names.add(f.__name__[3:])
  return wrapper

