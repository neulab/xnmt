"""
This implements events in XNMT. Events are handled globally, i.e. caller and handler
do not need to know about each other, and it is not possible to limit the scope of an
event. Event handling involves two parts:

- Registering an event caller. Callers are always class methods, and registration simply
  means decorating them as such:
  
  class MyObject(object):
    @register_xnmt_event
    def my_event():
      pass
  
  
- Defining event handlers. Event handlers are again always class methods. The following
  must hold:
  - handlers are named 'on_' + name of the event, e.g. on_my_event
  - method arguments must be consistent
  - handlers must be decorated with @handle_xnmt_event
  - the instances of the handler must call register_handler(self) in its __init__ method

  example:
  
  class AnotherObject(object):
    def __init__(self):
      register_handler(self)
    @handle_xnmt_event
    def on_my_event():
      # do something
  

  events can also return values. To make use of return   values, 2 special decorators are
  available:
  - @register_xnmt_event_assign assumes a special keyword argument named "context".
    The return value is an updated value for this context argument, and will then be
    passed on to the next event handler
  - @register_xnmt_event_sum here the return values of all handlers are summed up and
    returned.

"""

import inspect 

def clear():
  """
  clear handler (mostly useful to remove side effects when running a series of unit tests)
  """
  global handler_instances, handler_method_names, event_names
  handler_instances = []
  handler_method_names = set()
  event_names = set()

handler_instances = []
handler_method_names = set()
event_names = set()

def register_handler(inst):
  handler_instances.append(inst)
    
def register_xnmt_event(f):
  def wrapper(obj, *args, **kwargs):
    assert handler_method_names <= event_names, "detected handler for non-existant event: {}".format(handler_method_names-event_names)
    f(obj, *args, **kwargs)
    for handler in handler_instances:
      bound_handler = getattr(handler, "on_" + f.__name__, None)
      if bound_handler:
        ret = bound_handler(*args, **kwargs)
        if type(ret)!=tuple or len(ret)!=2 or ret[1][3:]!=f.__name__:
          raise RuntimeError("attempted to call unregistered handler {}".format(f.__name__))
  event_names.add(f.__name__)
  return wrapper
  
def register_xnmt_event_assign(f):
  def wrapper(obj, *args, **kwargs):
    assert "context" in kwargs, "register_xnmt_event_assign requires \"context\" in kwargs"
    assert handler_method_names <= event_names, "detected handler for non-existant event: {}".format(handler_method_names-event_names)
    kwargs["context"] = f(obj, *args, **kwargs)
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
  def wrapper(obj, *args, **kwargs):
    assert handler_method_names <= event_names, "detected handler for non-existant event: {}".format(handler_method_names-event_names)
    res = f(obj, *args, **kwargs)
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
  def wrapper(obj, *args, **kwargs):
    return f(obj, *args, **kwargs), f.__name__
  assert f.__name__.startswith("on_"), "xnmt event handlers must be named on_*, found {}".format(f.__name__)
  handler_method_names.add(f.__name__[3:])
  return wrapper
  
class GeneratorModel(object):
  def generate_output(self, *args, **kwargs):
    # Generate the output
    generation_output = self.generate(*args, **kwargs)
    # Post process it
    if hasattr(self, "post_processor"):
      generation_output = self.post_processor.process_outputs(generation_output)[0]
    return generation_output

  def generate(self, *args, **kwargs):
    raise NotImplementedError()

  @register_xnmt_event
  def initialize_generator(self, **kwargs):
    pass

  @register_xnmt_event
  def new_epoch(self):
    pass

  @register_xnmt_event
  def set_train(self, val):
    pass

  @register_xnmt_event
  def start_sent(self):
    pass

  def calc_loss(self, src, trg, src_mask=None, trg_mask=None):
    raise NotImplementedError()

  @register_xnmt_event_sum
  def calc_additional_loss(self, reward):
    ''' Calculate reinforce loss based on the reward
    :param reward: The default is log likelihood (-1 * calc_loss).
    '''
    return None

