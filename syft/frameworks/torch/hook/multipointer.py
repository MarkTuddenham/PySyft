"""Hook MultiPointer Tensor."""
from functools import wraps

from syft.frameworks.torch.tensors.interpreters import MultiPointerTensor
from syft.frameworks.torch.hook.hook_args import hook_method_args
from syft.frameworks.torch.hook.hook_args import hook_response


def hook_multi_pointer_tensor_methods(hook):
    """Create hook for MultiPointer methods.

    Add hooked version of all methods of the torch Tensor to the
    Multi Pointer tensor: instead of performing the native tensor
    method, it will be sent remotely for each pointer to the
    location it is pointing at.
    """
    tensor_type = hook.torch.Tensor
    # Use a pre-defined list to select the methods to overload
    for attr in hook.to_auto_overload[tensor_type]:
        if attr not in dir(MultiPointerTensor):
            new_method = hook.get_hooked_multi_pointer_method(attr)
            setattr(MultiPointerTensor, attr, new_method)


def get_hooked_multi_pointer_method(attr: str):
    """Update method to send itself to multiple remote workers.

    Args:
        attr: the method to hook

    Return:
        the hooked method
    """

    def dispatch(args, k):
        return map(lambda x: x[k] if isinstance(x, dict) else x, args)

    @wraps(attr)
    def overloaded_attr(self, *args, **kwargs):
        """Operate the hooking."""
        # Replace all syft tensor with their child attribute
        new_self, new_args, new_kwargs = hook_method_args(attr, self, args, kwargs)

        results = {}
        for k, v in new_self.items():
            results[k] = v.__getattribute__(attr)(*dispatch(new_args, k), **new_kwargs)

        # Put back MultiPointerTensor on the tensors found in the response
        response = hook_response(
            attr, results, wrap_type=MultiPointerTensor, wrap_args=self.get_class_attributes()
        )

        return response

    return overloaded_attr
