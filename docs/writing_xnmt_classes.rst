.. _sec-writing-classes:

Writing XNMT classes
====================

In order to write new components that can be created both from YAML config files as well as programmatically, support
sharing of network parameters, etc., one must adhere to the Serializable interface including a few simple conventions:

.. note:: XNMT will perform automatic checks and raise an informative error in case these conventions are violated,
  so there is no need to worry about these too much.

Marking classes as serializable
"""""""""""""""""""""""""""""""

Classes are marked as serializable by specifying :class:`xnmt.persistence.Serializable` as super class. They must
specify a unique yaml_tag class attribute, set to ``!ClassName`` with ClassName replaced by the class name. It follows
that class names must be unique, even across different XNMT modules.
(Note: Serializable should be explicitly specified even if another super class already does the same)

Specifying init arguments
"""""""""""""""""""""""""
The arguments accepted in the YAML config file correspond directly to the arguments of the class's ``__init__()``
method. The ``__init__`` is required to be decorated with ``@xnmt.persistence.serializable_init``.
Note that sub-objects are initialized before being passed to ``__init__``, and in the order in which they are
specified in ``__init__``.

Using trainable parameters
""""""""""""""""""""""""""

If the component uses trainable parameters, it must first request its own unique parameter collection using
``my_params = xnmt.param_collection.ParamManager.my_params(self)``. The ``my_params`` object should only be requested
and used within the ``__init__()`` method and never be passed to sub-objects that are ``Serializable``.
It is not possible to allocate parameters after ``__init__`` has returned.

Behind the scenes, components will get assigned a unique subcollection id which ensures
that they can be loaded later along with their pretrained weights, and even combined with components trained from
a different config file.

The syntax then depends on the backend:

DyNet backend:
``````

``my_params`` is an instance of ``dy.ParameterCollection``. Therefore, we can use ``my_params.add_parameters()`` etc.

Pytorch backend:
````````

``my_params`` is an instance of ``torch.nn.ModuleList``. To register trainable parameters, we there create a module
as ``mod = torch.nn.SomeModule()`` and then use ``my_params.append(mod)``.

Using Serializable subcomponents
""""""""""""""""""""""""""""""""
If a class uses helper objects that are also ``Serializable``, this must occur in a certain way:

 - the ``Serializable`` object must be accepted as argument in ``__init__``.
 - It can be set to ``None`` by default, in which case it must be constructed manually within ``__init__``.
   This should take place using the ``Serializable.add_serializable_component()`` helper, e.g. with the following idiom:

   .. code-block:: python

     @serializable_init
     def __init__(self, ..., vocab_projector=None, ...):
       ...
       self.vocab_projector = \
            self.add_serializable_component(\
                    "vocab_projector",
                    vocab_projector,
                    lambda: xnmt.linear.Linear(input_dim=mlp_hidden_dim,
                                               output_dim=vocab_size,
                                               param_init=param_init_output,
                                               bias_init=bias_init_output))
       ...
