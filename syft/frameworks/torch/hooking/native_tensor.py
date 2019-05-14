import inspect
import re
import logging
import types
import copy
import torch

# import torch.nn as nn
from functools import wraps


import syft
from syft.workers import BaseWorker

from syft.exceptions import route_method_exception
from syft.exceptions import TensorsNotCollocatedException

from syft.frameworks.torch.tensors.decorators import LoggingTensor
from syft.frameworks.torch.tensors.interpreters import TorchTensor
from syft.frameworks.torch.tensors.interpreters import PointerTensor
from syft.frameworks.torch.tensors.interpreters import FixedPrecisionTensor
from syft.frameworks.torch.tensors.interpreters.abstract import initialize_tensor
from syft.frameworks.torch.tensors.interpreters.abstract import _apply_args

from syft.frameworks.torch.hooking.additive_shared import hook_additive_shared_tensor_methods
from syft.frameworks.torch.hooking.multipointer import hook_multi_pointer_tensor_methods
from syft.frameworks.torch.hooking.torch_attributes import TorchAttributes


def hook_native_tensor(hook, tensor_type: type, syft_type: type):
    """Adds PySyft Tensor Functionality to the given native tensor type.

    Overloads the given native Torch tensor to add PySyft Tensor
    Functionality. Overloading involves modifying the tensor type with
    PySyft's added functionality. You may read about what kind of
    modifications are made in the methods that this method calls.

    Args:
        tensor_type: The type of tensor being hooked (in this refactor
            this is only ever torch.Tensor, but in previous versions of
            PySyft this iterated over all tensor types.
        syft_type: The abstract type whose methods should all be added to
            the tensor_type class. In practice this is always TorchTensor.
            Read more about it there.
    """
    # Reinitialize init method of Torch tensor with Syft init
    _add_registration_to___init__(hook, tensor_type, torch_tensor=True)

    # Overload Torch tensor properties with Syft properties
    _hook_properties(tensor_type)

    # Returns a list of methods to be overloaded, stored in the dict to_auto_overload
    # with tensor_type as a key
    hook.to_auto_overload[tensor_type] = _which_methods_should_we_auto_overload(tensor_type)

    # [We don't rename native methods as torch tensors are not hooked] Rename native functions
    # #self._rename_native_functions(tensor_type)

    # Overload auto overloaded with Torch methods
    _add_methods_from__torch_tensor(tensor_type, syft_type)

    _hook_native_methods(tensor_type)


def _which_methods_should_we_auto_overload(tensor_type: type):
    """Creates a list of Torch methods to auto overload.

        By default, it looks for the intersection between the methods of
        tensor_type and torch_type minus those in the exception list
        (syft.torch.exclude).

        Args:
            tensor_type: Iterate through the properties of this tensor type.
            syft_type: Iterate through all attributes in this type.

        Returns:
            A list of methods to be overloaded.
        """

    boolean_comparators = ["__gt__", "__ge__", "__lt__", "__le__"]

    to_overload = boolean_comparators

    for attr in dir(tensor_type):

        # Conditions for overloading the method
        if attr in syft.torch.exclude:
            continue
        if not hasattr(tensor_type, attr):
            continue

        lit = getattr(tensor_type, attr)
        is_base = attr in dir(object)
        is_desc = inspect.ismethoddescriptor(lit)
        is_func = isinstance(lit, types.FunctionType)
        try:
            is_service_func = "HookService" in lit.__qualname__
        except AttributeError:
            is_service_func = False
        is_overloaded = re.match("native*", attr) is not None

        if (is_desc or (is_func and not is_service_func)) and not is_base and not is_overloaded:
            to_overload.append(attr)

    return set(to_overload)


def _add_registration_to___init__(hook, tensor_type: type, torch_tensor: bool = False):
    """Adds several attributes to the tensor.

    Overloads tensor_type.__init__ to add several attributes to the tensor
    as well as (optionally) registering the tensor automatically.
    TODO: auto-registration is disabled at the moment, this might be bad.

    Args:
        tensor_type: The type of tensor being hooked (in this refactor this
            is only ever torch.Tensor, but in previous versions of PySyft
            this iterated over all tensor types.
        torch_tensor: An optional boolean parameter (default False) to
            specify whether to skip running the native initialization
            logic. TODO: this flag might never get used.
    """
    if "native___init__" not in dir(tensor_type):
        tensor_type.native___init__ = tensor_type.__init__

    def new___init__(cls, *args, owner=None, id=None, register=True, **kwargs):
        initialize_tensor(
            hook=hook, cls=cls, id=id, torch_tensor=torch_tensor, init_args=args, init_kwargs=kwargs
        )

    tensor_type.__init__ = new___init__


def _hook_properties(tensor_type: type):
    """Overloads tensor_type properties.

    This method gets called only on torch.Tensor. If you're not sure how
    properties work, read:
    https://www.programiz.com/python-programming/property

    Args:
        tensor_type: The tensor type which is having properties
            added to it, typically just torch.Tensor.
    """

    @property
    def location(self):
        return self.child.location

    tensor_type.location = location

    @property
    def id_at_location(self):
        return self.child.id_at_location

    tensor_type.id_at_location = id_at_location

    @property
    def id(self):
        if not hasattr(self, "_id"):
            self._id = syft.ID_PROVIDER.pop()
        return self._id

    @id.setter
    def id(self, new_id):
        self._id = new_id
        return self

    tensor_type.id = id

    @property
    def owner(self):
        if not hasattr(self, "_owner"):
            self._owner = syft.local_worker
        return self._owner

    @owner.setter
    def owner(self, new_owner):
        self._owner = new_owner
        return self

    tensor_type.owner = owner

    @property
    def is_wrapper(self):
        if not hasattr(self, "_is_wrapper"):
            self._is_wrapper = False
        return self._is_wrapper

    @is_wrapper.setter
    def is_wrapper(self, it_is_a_wrapper):
        self._is_wrapper = it_is_a_wrapper
        return self

    tensor_type.is_wrapper = is_wrapper

    tensor_type.native_shape = tensor_type.shape
    tensor_type.native_data = tensor_type.data


@staticmethod
def _add_methods_from__torch_tensor(tensor_type: type, syft_type: type):
    """Adds methods from the TorchTensor class to the native torch tensor.

    The class TorchTensor is a proxy to avoid extending directly the torch
    tensor class.

    Args:
        tensor_type: The tensor type to which we are adding methods
            from TorchTensor class.
    """
    exclude = [
        "__class__",
        "__delattr__",
        "__dir__",
        "__doc__",
        "__dict__",
        "__format__",
        "__getattribute__",
        "__hash__",
        "__init__",
        "__init_subclass__",
        "__weakref__",
        "__ne__",
        "__new__",
        "__reduce__",
        "__reduce_ex__",
        "__setattr__",
        "__sizeof__",
        "__subclasshook__",
        "_get_type",
        # "__eq__", # FIXME it now overwritten in native.py to use torch.eq, because of pb between == & __eq__ See #2030
        "__gt__",
        "__ge__",
        "__lt__",
        "__le__",
    ]
    # For all methods defined in TorchTensor which are not internal methods (like __class__etc)
    for attr in dir(syft_type):
        if attr not in exclude:
            if hasattr(tensor_type, attr):
                setattr(tensor_type, f"native_{attr}", getattr(tensor_type, attr))
            # Add to the native tensor this method
            setattr(tensor_type, attr, getattr(TorchTensor, attr))


def _hook_native_methods(tensor_type: type):
    """
    Add hooked version of all methods of to_auto_overload[tensor_type]
    to the tensor_type; instead of performing the native tensor
    method, the hooked version will be called

    Args:
        tensor_type: the tensor_type which holds the methods
    """
    # # Add methods defined in the TorchTensor class to the Pointer class
    # self._add_methods_from__torch_tensor(PointerTensor, TorchTensor)

    # Use a pre-defined list to select the methods to overload
    for attr in self.to_auto_overload[tensor_type]:
        # if we haven't already overloaded this function
        if f"native_{attr}" not in dir(tensor_type):
            native_method = getattr(tensor_type, attr)
            setattr(tensor_type, f"native_{attr}", native_method)
            new_method = self.get_hooked_method(attr)
            setattr(tensor_type, attr, new_method)


def get_hooked_method(hook_self, method_name):
    """
    Hook a method in order to replace all args/kwargs syft/torch tensors with
    their child attribute if they exist
    If so, forward this method with the new args and new self, get response
    and "rebuild" the torch tensor wrapper upon all tensors found
    If not, just execute the native torch method

    Args:
        attr (str): the method to hook
    Return:
        the hooked method
    """

    @wraps(method_name)
    def overloaded_native_method(self, *args, **kwargs):
        """
        Operate the hooking
        """

        if not hasattr(self, "child"):  # means that it's not a wrapper
            method = getattr(self, f"native_{method_name}")
            # Run the native function with the new args

            try:
                if isinstance(args, tuple):
                    response = method(*args, **kwargs)
                else:
                    response = method(args, **kwargs)

            except BaseException as e:
                # we can make some errors more descriptive with this method
                raise route_method_exception(e, self, args, kwargs)

        else:  # means that there is a wrapper to remove
            try:
                # Replace all torch tensor with their child attribute
                new_self, new_args, new_kwargs = syft.frameworks.torch.hook_args.hook_method_args(
                    method_name, self, args, kwargs
                )
            except BaseException as e:
                # we can make some errors more descriptive with this method
                raise route_method_exception(e, self, args, kwargs)

            # Send the new command to the appropriate class and get the response
            method = getattr(new_self, method_name)
            response = method(*new_args, **new_kwargs)

            # For inplace methods, just directly return self
            if syft.torch.is_inplace_method(method_name):
                return self

            # Put back the wrappers where needed
            response = syft.frameworks.torch.hook_args.hook_response(
                method_name, response, wrap_type=type(self), new_self=self
            )

        return response

    return overloaded_native_method
