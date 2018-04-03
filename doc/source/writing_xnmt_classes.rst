Writing XNMT classes
====================

In order to write new components that can be created both from YAML config files as well as programmatically, support
sharing of DyNet parameters, etc., one must adhere to the Serializable interface including a few simple conventions.

1. Classes must have ``Serializable`` from the ``xnmt.serialize.serializable`` module as super class.
2. Classes must have a unique name, regardless of module placement.
3. Class must specify a yaml_tag class attribute, set to ``!ClassName`` with ClassName replaced by the unique class
   name.
4. The ``__init__`` must be decorated with ``@serializable_init`` from the ``xnmt.serialize.serializer`` module. Even if
   ``__init__`` contains no functionality, it should be specified regardless.
5. In the YAML config file, all arguments given in the ``__init__`` method are accepted. Sub-objects are initialized
   before being passed to ``__init__``, and in the order in which they are specified in ``__init__``.
6. If the component uses DyNet parameters, the calls to ``dynet_model.add_parameters()`` etc. must take place in ``__init__`` (or a
   helper called from within ``__init__``). It is not permitted to allocate parameters after ``__init__`` has been executed.
   The component will get assigned its own unique DyNet parameter collection, which can be requested using
   ``xnmt.param_collection.ParamManager.my_params(self)``. Subcollections should never be passed to sub-objects
   that are ``Serializable``. Behind the scenes, components will get assigned a unique subcollection id which ensures
   that they can be loaded later along with their pretrained weights, and even combined with components trained from
   a different config file.
7. If a class uses helper objects that are also ``Serializable``, this must occur in a certain way:

 - the ``Serializable`` object must be accepted as argument in ``__init__``.
 - It can be set to ``None`` by default, in which case it must be constructed manually within ``__init__``.
   This should take place using the Serializable.add_serializable_component() helper, e.g. as follows:
   ``self.vocab_projector = self.add_serializable_component("vocab_projector", vocab_projector, lambda: xnmt.linear.Linear(input_dim=mlp_hidden_dim, output_dim=vocab_size, param_init=param_init_output, bias_init=bias_init_output))``
