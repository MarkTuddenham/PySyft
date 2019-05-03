"""Hook AdditiveSharingTensor Tensor."""
from functools import wraps

from typing import Callable, Any

from syft.frameworks.torch.tensors.interpreters import AdditiveSharingTensor

from .hook_args import hook_method_args
from .hook_args import hook_response


def hook_additive_shared_tensor_methods(hook):
    """Create hook for AdditiveSharingTensor methods.

    Add hooked version of all methods of the torch Tensor to the
    Additive Shared tensor: instead of performing the native tensor
    method, it will be forwarded to each share when it is relevant
    """

    tensor_type = hook.torch.Tensor
    # Use a pre-defined list to select the methods to overload
    for attr in hook.to_auto_overload[tensor_type]:
        if attr not in dir(AdditiveSharingTensor):
            new_method = get_hooked_additive_shared_method(attr)
            setattr(AdditiveSharingTensor, attr, new_method)


def get_hooked_additive_shared_method(attr: Callable[..., Any]):
    """Update a method to send it multiple remote workers.

    Args:
        attr: the method to hook

    Return:
        the hooked method
    """

    def dispatch(args, k):
        return map(lambda x: x[k] if isinstance(x, dict) else x, args)

    @wraps(attr)
    def overloaded_attr(self, *args, **kwargs):
        """Define the new attribute."""

        # Replace all syft tensor with their child attribute
        new_self, new_args, new_kwargs = hook_method_args(attr, self, args, kwargs)

        results = {}
        for k, v in new_self.items():
            results[k] = v.__getattribute__(attr)(*dispatch(new_args, k), **new_kwargs)

        # Put back AdditiveSharingTensor on the tensors found in the response
        response = hook_response(
            attr, results, wrap_type=AdditiveSharingTensor, wrap_args=self.get_class_attributes()
        )

        return response

    return overloaded_attr
