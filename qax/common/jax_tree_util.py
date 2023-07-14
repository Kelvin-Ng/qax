#from typing import (Any, Callable, Hashable, Iterable, NamedTuple, Optional, TypeVar)
#import operator as op
#from jax._src.lib import pytree
#from jax.tree_util import *
from dataclasses import dataclass
from jax._src.tree_util import *
from jax._src.tree_util import _registry

#T = TypeVar("T")
#U = TypeVar("U", bound=type[Any])
#
#


@dataclass(frozen=True)
class SequenceKey():
  idx: int
  def __str__(self):
    return f'[{repr(self.idx)}]'

@dataclass(frozen=True)
class DictKey():
  key: Hashable
  def __str__(self):
    return f'[{repr(self.key)}]'


@dataclass(frozen=True)
class FlattenedIndexKey():
  key: int
  def __str__(self):
    return f'[<flat index {self.key}>]'


KeyEntry = TypeVar("KeyEntry", bound=Hashable)
KeyPath = tuple[KeyEntry, ...]
#
#
_AuxData = TypeVar("_AuxData", bound=Hashable)
#
#
#PyTreeDef = pytree.PyTreeDef


class _RegistryWithKeypathsEntry(NamedTuple):
  flatten_with_keys: Callable[..., Any]
  unflatten_func: Callable[..., Any]


def _register_keypaths(
    ty: type[T], handler: Callable[[T], tuple[KeyEntry, ...]]
) -> None:
  def flatten_with_keys(xs):
    children, treedef = _registry[ty].to_iter(xs)
    return list(zip(handler(xs), children)), treedef
  if ty in _registry:
    _registry_with_keypaths[ty] = _RegistryWithKeypathsEntry(
        flatten_with_keys, _registry[ty].from_iter
    )


_registry_with_keypaths = {}


_register_keypaths(
    tuple, lambda xs: tuple(SequenceKey(i) for i in range(len(xs)))
)
_register_keypaths(
    list, lambda xs: tuple(SequenceKey(i) for i in range(len(xs)))
)
_register_keypaths(dict, lambda xs: tuple(DictKey(k) for k in sorted(xs)))

_register_keypaths(
    collections.defaultdict, lambda x: tuple(DictKey(k) for k in x.keys())
)

_register_keypaths(
    collections.OrderedDict, lambda x: tuple(DictKey(k) for k in x.keys())
)



def register_pytree_with_keys(
    nodetype: type[T],
    flatten_with_keys: Callable[
        [T], tuple[Iterable[tuple[KeyEntry, Any]], _AuxData]
    ],
    unflatten_func: Callable[[_AuxData, Iterable[Any]], T],
    flatten_func: Optional[
        Callable[[T], tuple[Iterable[Any], _AuxData]]
    ] = None,
):
  """Extends the set of types that are considered internal nodes in pytrees.

  This is a more powerful alternative to ``register_pytree_node`` that allows
  you to access each pytree leaf's key path when flattening and tree-mapping.

  Args:
    nodetype: a Python type to treat as an internal pytree node.
    flatten_with_keys: a function to be used during flattening, taking a value
      of type ``nodetype`` and returning a pair, with (1) an iterable for tuples
      of each key path and its child, and (2) some hashable auxiliary data to be
      stored in the treedef and to be passed to the ``unflatten_func``.
    unflatten_func: a function taking two arguments: the auxiliary data that was
      returned by ``flatten_func`` and stored in the treedef, and the
      unflattened children. The function should return an instance of
      ``nodetype``.
    flatten_func: an optional function similar to ``flatten_with_keys``, but
      returns only children and auxiliary data. It must return the children
      in the same order as ``flatten_with_keys``, and return the same aux data.
      This argument is optional and only needed for faster traversal when
      calling functions without keys like ``tree_map`` and ``tree_flatten``.
  """
  if not flatten_func:
    def flatten_func_impl(tree):
      key_children, treedef = flatten_with_keys(tree)
      return [c for _, c in key_children], treedef
    flatten_func = flatten_func_impl

  register_pytree_node(nodetype, flatten_func, unflatten_func)
  _registry_with_keypaths[nodetype] = _RegistryWithKeypathsEntry(
      flatten_with_keys, unflatten_func
  )


def register_pytree_with_keys_class(cls: U) -> U:
  """Extends the set of types that are considered internal nodes in pytrees.

  This function is similar to ``register_pytree_node_class``, but requires a
  class that defines how it could be flattened with keys.

  It is a thin wrapper around ``register_pytree_with_keys``, and
  provides a class-oriented interface::

    @register_pytree_with_keys_class
    class Special:
      def __init__(self, x, y):
        self.x = x
        self.y = y
      def tree_flatten_with_keys(self):
        return (((GetAttrKey('x'), self.x), (GetAttrKey('y'), self.y)), None)
      @classmethod
      def tree_unflatten(cls, aux_data, children):
        return cls(*children)
  """
  flatten_func = (
      op.methodcaller("tree_flatten") if hasattr(cls, "tree_flatten") else None
  )
  register_pytree_with_keys(
      cls, op.methodcaller("tree_flatten_with_keys"), cls.tree_unflatten,
      flatten_func
  )
  return cls


def tree_flatten_with_path(
    tree: Any, is_leaf: Optional[Callable[[Any], bool]] = None
) -> tuple[list[tuple[KeyPath, Any]], PyTreeDef]:
  """Flattens a pytree like ``tree_flatten``, but also returns each leaf's key path.

  Args:
    tree: a pytree to flatten. If it contains a custom type, it must be
      registered with ``register_pytree_with_keys``.
  Returns:
    A pair which the first element is a list of key-leaf pairs, each of
    which contains a leaf and its key path. The second element is a treedef
    representing the structure of the flattened tree.
  """
  _, tree_def = tree_flatten(tree, is_leaf)
  return _generate_key_paths(tree, is_leaf), tree_def


def tree_leaves_with_path(
    tree: Any, is_leaf: Optional[Callable[[Any], bool]] = None
) -> list[tuple[KeyPath, Any]]:
  """Gets the leaves of a pytree like ``tree_leaves`` and returns each leaf's key path.

  Args:
    tree: a pytree. If it contains a custom type, it must be registered with
      ``register_pytree_with_keys``.
  Returns:
    A list of key-leaf pairs, each of which contains a leaf and its key path.
  """
  return _generate_key_paths(tree, is_leaf)


def generate_key_paths(
    tree: Any, is_leaf: Optional[Callable[[Any], bool]] = None
) -> list[tuple[KeyPath, Any]]:
  return list(_generate_key_paths_((), tree, is_leaf))
_generate_key_paths = generate_key_paths  # alias for backward compat


# The overall logic should be same as PyTreeDef::FlattenIntoImpl
def _generate_key_paths_(
    key_path: KeyPath,
    tree: Any,
    is_leaf: Optional[Callable[[Any], bool]] = None,
) -> Iterable[tuple[KeyPath, Any]]:
  if is_leaf and is_leaf(tree):
    yield key_path, tree
    return
  key_handler = _registry_with_keypaths.get(type(tree))
  handler = _registry.get(type(tree))
  if key_handler:
    key_children, _ = key_handler.flatten_with_keys(tree)
    for k, c in key_children:
      yield from _generate_key_paths_((*key_path, k), c, is_leaf)
  elif handler:
    children, _ = handler.to_iter(tree)
    for i, c in enumerate(children):
      k = FlattenedIndexKey(i)
      yield from _generate_key_paths_((*key_path, k), c, is_leaf)
  elif isinstance(tree, tuple) and hasattr(tree, '_fields'):
    # handle namedtuple as a special case, based on heuristic
    key_children = [(GetAttrKey(s), getattr(tree, s)) for s in tree._fields]
    for k, c in key_children:
      print('k, c:', k, c)
      yield from _generate_key_paths_(tuple((*key_path, k)), c, is_leaf)
  else:
    yield key_path, tree  # strict leaf type


def tree_map_with_path(f: Callable[..., Any],
                       tree: Any, *rest: Any,
                       is_leaf: Optional[Callable[[Any], bool]] = None) -> Any:
  """Maps a multi-input function over pytree key path and args to produce a new pytree.

  This is a more powerful alternative of ``tree_map`` that can take the key path
  of each leaf as input argument as well.

  Args:
    f: function that takes ``2 + len(rest)`` arguments, aka. the key path and
      each corresponding leaves of the pytrees.
    tree: a pytree to be mapped over, with each leaf's key path as the first
      positional argument and the leaf itself as the second argument to ``f``.
    *rest: a tuple of pytrees, each of which has the same structure as ``tree``
      or has ``tree`` as a prefix.

  Returns:
    A new pytree with the same structure as ``tree`` but with the value at each
    leaf given by ``f(kp, x, *xs)`` where ``kp`` is the key path of the leaf at
    the corresponding leaf in ``tree``, ``x`` is the leaf value and ``xs`` is
    the tuple of values at corresponding nodes in ``rest``.
  """

  keypath_leaves, treedef = tree_flatten_with_path(tree, is_leaf)
  keypath_leaves = list(zip(*keypath_leaves))
  all_keypath_leaves = keypath_leaves + [treedef.flatten_up_to(r) for r in rest]
  return treedef.unflatten(f(*xs) for xs in zip(*all_keypath_leaves))
