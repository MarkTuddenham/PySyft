import inspect
import re
import logging
import types
import copy
import torch

# import torch.nn as nn
from functools import wraps


import syft

from syft import workers

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
from syft.frameworks.torch.hooking.native_tensor import hook_native_tensor
from syft.frameworks.torch.hooking.torch_attributes import TorchAttributes


class TorchHook:
    """A Hook which Overrides Methods on PyTorch Tensors.

    The purpose of this class is to:
        * extend torch methods to allow for the moving of tensors from one
        worker to another.
        * override torch methods to execute commands on one worker that are
        called on tensors controlled by the local worker.

    This class is typically the first thing you will initialize when using
    PySyft with PyTorch because it is responsible for augmenting PyTorch with
    PySyft's added functionality (such as remote execution).

    Args:
        local_worker: An optional BaseWorker instance that lets you provide a
            local worker as a parameter which TorchHook will assume to be the
            worker owned by the local machine. If you leave it empty,
            TorchClient will automatically initialize a
            :class:`.workers.VirtualWorker` under the assumption you're looking
            to do local experimentation or development.
        is_client: An optional boolean parameter (default True), indicating
            whether TorchHook is being initialized as an end-user client.This
            can impact whether or not variables are deleted when they fall out
            of scope. If you set this incorrectly on a end user client, Tensors
            and Variables will never be deleted. If you set this incorrectly on
            a remote machine (not a client), tensors will not get saved. It's
            really only important if you're not initializing the local worker
            yourself.
        verbose: An optional boolean parameter (default True) to indicate
            whether or not to print the operations as they occur.
        queue_size: An integer optional parameter (default 0) to specify the
            max length of the list that stores the messages to be sent.

    Example:
        >>> import syft as sy
        >>> hook = sy.TorchHook()
        Hooking into Torch...
        Overloading Complete.
        >>> x = sy.Tensor([-2,-1,0,1,2,3])
        >>> x
        -2
        -1
        0
        1
        2
        3
        [syft.core.frameworks.torch.tensor.FloatTensor of size 6]
    """

    def __init__(
        self,
        torch=torch,
        local_worker: BaseWorker = None,
        is_client: bool = True,
        verbose: bool = True,
    ) -> None:
        """Initialize the hook.

        Initialize the hook and define all the attributes pertaining to the
        torch hook in a special TorchAttribute class, that will be added in the
        syft.torch attributes. Hence, this parameters are now conveyed by the
        syft module.
        """
        # Save the provided torch module as an attribute of the hook
        self.torch = torch

        # Save the local worker as an attribute
        self.local_worker = local_worker

        if hasattr(torch, "torch_hooked"):
            logging.warning("Torch was already hooked... skipping hooking process")
            self.local_worker = syft.local_worker
            return
        else:
            torch.torch_hooked = True

        # Add all the torch attributes in the syft.torch attr
        syft.torch = TorchAttributes(torch, self)

        if self.local_worker is None:
            # Every TorchHook instance should have a local worker which is
            # responsible for interfacing with other workers. The worker
            # interface is what allows the Torch specific code in TorchHook to
            # be agnostic to the means by which workers communicate (such as
            # peer-to-peer, sockets, through local ports, or all within the
            # same process)
            self.local_worker = workers.VirtualWorker(
                hook=self, is_client_worker=is_client, id="me"
            )
        else:
            self.local_worker.hook = self

        self.to_auto_overload = {}

        self.args_hook_for_overloaded_attr = {}

        hook_native_tensor(self, torch.Tensor, TorchTensor)

        # Add all hooked tensor methods to pointer but change behaviour to have the cmd sent
        self._hook_pointer_tensor_methods()

        # Add all hooked tensor methods to AdditiveSharingTensor tensor but change behaviour
        # to all shares (when it makes sense, otherwise the method is overwritten in the
        # AdditiveSharingTensor class)
        hook_additive_shared_tensor_methods(self)

        # Add all hooked tensor methods to multi_pointer to change behaviour to have the cmd
        # sent to all child pointers.
        hook_multi_pointer_tensor_methods(self)

        # Add all hooked tensor methods to Logging tensor but change behaviour to just forward
        # the cmd to the next child (behaviour can be changed in the SyftTensor class file)
        self._hook_syft_tensor_methods(LoggingTensor)

        # Add all hooked tensor methods to FixedPrecisionTensor tensor but change behaviour
        # to just forward the cmd to the next child (behaviour can be changed in the
        # SyftTensor class file)
        self._hook_syft_tensor_methods(FixedPrecisionTensor)

        # Hook the tensor constructor function
        self._hook_tensor()

        # Hook the Parameter methods to store tensor chains in parameters
        self._hook_parameters()

        # Hook torch functions from modules like torch.add OR torch.nn.functional (containing relu, etc.)
        self._hook_torch_module()

        # Hook torch.nn (containing Linear and Convolution layers)
        self._hook_module()

        # Add the local_worker to syft so that it can be found if the hook is
        # called several times
        syft.local_worker = self.local_worker
        syft.hook = self

    def _hook_syft_tensor_methods(self, syft_type: type):
        """
        Add hooked version of all methods of to_auto_overload[tensor_type]
        to the syft_type, so that they act like regular tensors in
        terms of functionality, but instead of performing the native tensor
        method, it will be forwarded to each share when it is relevant

        Args:
            syft_type: the syft_type which holds the methods
        """

        tensor_type = self.torch.Tensor
        # Use a pre-defined list to select the methods to overload
        for attr in self.to_auto_overload[tensor_type]:
            if attr not in dir(syft_type):
                new_method = self.get_hooked_syft_method(attr)
                setattr(syft_type, attr, new_method)

    def _hook_pointer_tensor_methods(self):
        """
        Add hooked version of all methods of the torch Tensor to the
        Pointer tensor: instead of performing the native tensor
        method, it will be sent remotely to the location the pointer
        is pointing at.
        """

        boolean_comparators = ["__gt__", "__ge__", "__lt__", "__le__"]

        tensor_type = self.torch.Tensor
        # Use a pre-defined list to select the methods to overload
        for attr in self.to_auto_overload[tensor_type]:
            if attr not in dir(PointerTensor) or attr in boolean_comparators:
                new_method = self.get_hooked_pointer_method(attr)
                setattr(PointerTensor, attr, new_method)

    def _hook_parameters(self):
        """
        This method overrides the torch Parameter class such that
        it works correctly with our overridden tensor types. The
        native torch Parameter class kept deleting all of our
        attributes on our custom tensors, so we wrote our own.
        """

        # Hook __new__ to handle when non-pure torch tensors are given as data attribute

        def hooked__new__(cls, data=None, requires_grad=True):
            if data is None:
                data = torch.Tensor()

            # If data is not a pure torch tensor you need to store the chain in a
            # specific place otherwise it will get deleted
            if not isinstance(data, torch.Tensor) or hasattr(data, "child"):
                p = torch.Tensor._make_subclass(cls, torch.Tensor(), requires_grad)
                if isinstance(data, torch.Tensor):  # so it's a wrapper: remove it
                    p.child = data.child
                else:
                    p.child = data
            else:
                p = torch.Tensor._make_subclass(cls, data, requires_grad)

            return p

        torch.nn.Parameter.__new__ = hooked__new__

        # Hook __repr__ to handle chain repr when needed

        torch.nn.Parameter.native_param___repr__ = torch.nn.Parameter.__repr__

        def hooked__repr__(self):
            if hasattr(self, "child"):
                return "Parameter containing:\n" + self.child.__repr__()
            else:
                return self.native_param___repr__()

        # torch.nn.Parameter.__repr__ = hooked__repr__

        # Hook .data to handle chain assignment when needed

        torch.nn.Parameter.native_param_data = torch.nn.Parameter.data

        @property
        def data(self):

            if hasattr(self, "child"):
                to_return = self.child.attr("data")
            else:

                to_return = self.native_param_data

                # good to ensure that the ID stays consistent
                # not 100% this is required but it's at least
                # good practice
                try:
                    to_return.id = self.data_id
                except:
                    self.data_id = to_return.id

            return to_return

        @data.setter
        def data(self, new_data):

            # If data is not a pure torch tensor you need to store the chain in a
            # specific place otherwise it will get deleted
            if not isinstance(new_data, torch.Tensor) or hasattr(new_data, "child"):
                self.child = new_data  # .wrap()
            else:
                if hasattr(self, "child"):
                    del self.child

                self.native_param_data.set_(new_data)  # .wrap()
            return self

        torch.nn.Parameter.data = data

        # Hook .grad to handle chain assignment when needed

        torch.nn.Parameter.native_param_grad = torch.nn.Parameter.grad

        @property
        def grad(self):

            if hasattr(self, "child"):
                to_return = self.child.attr("grad")
                if isinstance(to_return.child, syft.PointerTensor):
                    if to_return.child.is_none():
                        to_return = None
            else:
                to_return = self.native_param_grad

                # good to ensure that the ID stays consistent
                # not 100% this is required but it's at least
                # good practice
                try:
                    to_return.id = self.grad_id
                except AttributeError:
                    if to_return is not None and hasattr(to_return, "id"):
                        self.grad_id = to_return.id

            return to_return

        @grad.setter
        def grad(self, new_grad):

            # If grad is not a pure torch tensor you need to store the chain in a
            # specific place otherwise it will get deleted
            if new_grad is not None and (
                not isinstance(new_grad, torch.Tensor) or hasattr(new_grad, "child")
            ):
                self.child.grad = new_grad  # .wrap()
            else:
                if self.native_param_grad is not None:
                    self.native_param_grad.set_(new_grad)  # .wrap()
                elif new_grad is not None:
                    self.native_param_grad = new_grad
            return self

        torch.nn.Parameter.grad = grad

    def _hook_torch_module(self):
        """Overloads functions in the main torch modules.
        The way this is accomplished is by first moving all existing module
        functions in the torch module to native_<function_name_here>.

        Example:
            the real :func:`torch.cat` will become :func:`torch.native_cat`
            and :func:`torch.cat` will have our hooking code.
        """

        def perform_overloading(torch_module, func):

            # Where the overloading happens
            # 1. Get native function
            native_func = getattr(torch_module, func)
            # 2. Check it is a proper function
            if type(native_func) in [types.FunctionType, types.BuiltinFunctionType]:
                # 3. Build the hooked function
                new_func = self.get_hooked_func(native_func)
                # 4. Move the native function
                setattr(torch_module, f"native_{func}", native_func)
                # 5. Put instead the hooked one
                setattr(torch_module, func, new_func)

        # Hard fix for PyTorch versions < 1.0.2
        syft.torch.apply_fix16922(self.torch)

        torch_modules = syft.torch.torch_modules

        for module_name, torch_module in torch_modules.items():
            for func in dir(torch_module):

                # Some functions we want to ignore (not override). Such functions have been hard
                # coded into the torch_attribute exclude (see TorchAttribute class)
                if func in syft.torch.exclude:
                    continue

                # ignore dunder functions
                if "__" in func:
                    continue

                # ignore capitalized func values which are Classes not functinos
                if func[0].isupper():
                    continue

                # ignore hidden functins
                if func[0] == "_":
                    continue

                # If we haven't already overloaded this function
                if "native_" in func or f"native_{func}" in dir(torch_module):
                    continue

                perform_overloading(torch_module, func)

    def get_hooked_pointer_method(hook_self, attr):
        """
        Hook a method to send it to remote worker

        Args:
            attr (str): the method to hook
        Return:
            the hooked method
        """

        @wraps(attr)
        def overloaded_pointer_method(self, *args, **kwargs):
            """
            Operate the hooking
            """
            pointer = self
            # Get info on who needs to send where the command
            owner = pointer.owner
            location = pointer.location

            if len(args) > 0:
                if isinstance(args[0], PointerTensor):
                    if args[0].location.id != location.id:
                        raise TensorsNotCollocatedException(pointer, args[0], attr)

            # Send the command
            command = (attr, self, args, kwargs)

            response = owner.send_command(location, command)

            return response

        return overloaded_pointer_method

    def get_hooked_syft_method(hook_self, attr):
        """
        Hook a method in order to replace all args/kwargs syft/torch tensors with
        their child attribute, forward this method with the new args and new self,
        get response and "rebuild" the syft tensor wrapper upon all tensors found

        Args:
            attr (str): the method to hook
        Return:
            the hooked method
        """

        @wraps(attr)
        def overloaded_syft_method(self, *args, **kwargs):
            """
            Operate the hooking
            """
            # TODO: I can't manage the import issue, can you?
            # Replace all syft tensor with their child attribute
            new_self, new_args, new_kwargs = syft.frameworks.torch.hook_args.hook_method_args(
                attr, self, args, kwargs
            )

            # Send it to the appropriate class and get the response
            response = getattr(new_self, attr)(*new_args, **new_kwargs)

            # Put back SyftTensor on the tensors found in the response
            response = syft.frameworks.torch.hook_args.hook_response(
                attr, response, wrap_type=type(self), wrap_args=self.get_class_attributes()
            )

            return response

        return overloaded_syft_method

    def get_hooked_func(hook_self, attr):
        """
        Hook a function in order to inspect its args and search for pointer
        or other syft tensors.
        - Calls to this function with normal tensors or numbers / string trigger
          usual behaviour
        - Calls with pointers send the command to the location of the pointer(s)
        - Calls with syft tensor will in the future trigger specific behaviour

        Args:
            attr (str): the method to hook
        Return:
            the hooked method
        """

        if attr.__module__ is None:
            attr.__module__ = "torch"

        @wraps(attr)
        def overloaded_func(*args, **kwargs):
            """
            Operate the hooking
            """
            cmd_name = f"{attr.__module__}.{attr.__name__}"
            command = (cmd_name, None, args, kwargs)
            response = TorchTensor.handle_func_command(command)
            return response

        return overloaded_func

    def _hook_tensor(hook_self):
        """Hooks the function torch.tensor()
        We need to do this separately from hooking the class because internally
        torch does not pick up the change to add the args
        Args:
            hook_self: the hook itself
        """

        if "native_tensor" not in dir(hook_self.torch):
            hook_self.torch.native_tensor = hook_self.torch.tensor

        def new_tensor(*args, owner=None, id=None, register=True, **kwargs):
            current_tensor = hook_self.torch.native_tensor(*args, **kwargs)
            _apply_args(hook_self, current_tensor, owner, id)
            if register:
                current_tensor.owner.register_obj(current_tensor)

            return current_tensor

        hook_self.torch.tensor = new_tensor

    def _hook_module(self):
        """Overloading torch.nn.Module with PySyft functionality, the primary module
           responsible for core ML functionality such as Neural network layers and
           loss functions.
           It is important to note that all the operations are actually in-place.
        """

        def module_is_missing_grad(model):
            """Checks if all the parameters in the model have been assigned a gradient"""
            for p in model.parameters():
                if p.grad is None:
                    return True
            return False

        def create_grad_objects(model):
            """Assigns gradient to model parameters if not assigned"""
            # for p in model.parameters():
            #     o = p.sum()
            #     o.backward()
            #     p.grad -= p.grad

        def module_send_(nn_self, dest):
            """Overloads torch.nn instances so that they could be sent to other workers"""

            if module_is_missing_grad(nn_self):
                create_grad_objects(nn_self)

            for p in nn_self.parameters():
                p.send_(dest)

            return nn_self

        self.torch.nn.Module.send = module_send_

        def module_move_(nn_self, destination):

            params = list(nn_self.parameters())
            for p in params:
                p.child.wrap().move(destination)

        self.torch.nn.Module.move = module_move_

        # def module_end_get_(nn_self):
        #     """Overloads send to remote for torch.nn.Module."""
        #     if module_is_missing_grad(nn_self):
        #         create_grad_objects(nn_self)
        #
        #     for p in nn_self.parameters():
        #         p.end_get()
        #
        #     return nn_self
        #
        # self.torch.nn.Module.end_get = module_end_get_
        #
        # def module_move_(nn_self, dest):
        #     return nn_self.send(dest).end_get()
        #
        # self.torch.nn.Module.move = module_move_

        def module_get_(nn_self):
            """overloads torch.nn instances with get method so that parameters could be sent back to owner"""
            for p in nn_self.parameters():
                p.get_()

            return nn_self

        self.torch.nn.Module.get = module_get_

        def module_share_(nn_self, *args, **kwargs):
            """Overloads fix_precision for torch.nn.Module."""
            # TODO: add .data and .grad to syft tensors
            # if module_is_missing_grad(nn_self):
            #    create_grad_objects(nn_self)

            for p in nn_self.parameters():
                p.share_(*args, **kwargs)

            return nn_self

        self.torch.nn.Module.share = module_share_

        def module_fix_precision_(nn_self, *args, **kwargs):
            """Overloads fix_precision for torch.nn.Module."""
            if module_is_missing_grad(nn_self):
                create_grad_objects(nn_self)

            for p in nn_self.parameters():
                p.fix_precision_(*args, **kwargs)

            return nn_self

        self.torch.nn.Module.fix_precision = module_fix_precision_

        def module_float_precision_(nn_self):
            """Overloads float_precision for torch.nn.Module, convert fix_precision
            parameters to normal float parameters"""
            # TODO: add .data and .grad to syft tensors
            # if module_is_missing_grad(nn_self):
            #    create_grad_objects(nn_self)

            for p in nn_self.parameters():
                p.float_precision_()

            return nn_self

        self.torch.nn.Module.float_precision = module_float_precision_

        def module_copy_(nn_self):
            return copy.deepcopy(nn_self)

        self.torch.nn.Module.copy = module_copy_
