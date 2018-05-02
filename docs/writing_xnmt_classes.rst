.. _sec-writing-classes:

Writing XNMT classes
====================

In order to write new components that can be created both from YAML config files as well as programmatically, support
sharing of DyNet parameters, etc., one must adhere to the Serializable interface including a few simple conventions:

.. note:: XNMT will perform automatic checks and raise an informative error in case these conventions are violated,
  so there is no need to worry about these too much.

Marking classes as serializable
"""""""""""""""""""""""""""""""

Classes are marked as serializable by specifying :class:`xnmt.persistence.Serializable` as super class.
They must specify a unique yaml_tag class attribute, set to ``!ClassName`` with ClassName replaced by the class
name. It follows that class names must be unique, even across different XNMT modules.

Specifying init arguments
"""""""""""""""""""""""""
The arguments accepted in the YAML config file correspond directly to the arguments of the class's ``__init__()``
method. The ``__init__`` is required to be decorated with ``@xnmt.persistence.serializable_init``.
Note that sub-objects are initialized before being passed to ``__init__``, and in the order in which they are
specified in ``__init__``.

Using DyNet parameters
""""""""""""""""""""""
If the component uses DyNet parameters, the calls to ``dynet_model.add_parameters()`` etc. must take place in
``__init__`` (or a helper called from within ``__init__``). It is not possible to allocate parameters after
``__init__`` has returned.
The component will get assigned its own unique DyNet parameter collection, which can be requested using
``xnmt.param_collection.ParamManager.my_params(self)``. Subcollections should never be passed to sub-objects
that are ``Serializable``. Behind the scenes, components will get assigned a unique subcollection id which ensures
that they can be loaded later along with their pretrained weights, and even combined with components trained from
a different config file.

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
