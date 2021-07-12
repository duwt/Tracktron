import itertools
from typing import Any, Dict, List, Union
import torch


class TrackletStates:
    """
    Modified from `detectron2.structures.Instances`

    This class represents a list of tracklet states.
    """

    def __init__(self, **kwargs: Any):
        """
        Args:
            kwargs: fields to add to this `TrackletStates`.
        """
        self._fields: Dict[str, Any] = {}
        for k, v in kwargs.items():
            self.set(k, v)

    def __setattr__(self, name: str, val: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, val)
        else:
            self.set(name, val)

    def __getattr__(self, name: str) -> Any:
        if name == "_fields" or name not in self._fields:
            raise AttributeError("Cannot find field '{}' in the given TrackletStates!".format(name))
        return self._fields[name]

    def set(self, name: str, value: Any) -> None:
        """
        Set the field named `name` to `value`.
        The length of `value` must be the number of tracklets,
        and must agree with other existing fields in this object.
        """
        data_len = len(value)
        if len(self._fields):
            assert (
                len(self) == data_len
            ), "Adding a field of length {} to a TrackletStates of length {}".format(data_len, len(self))
        self._fields[name] = value

    def has(self, name: str) -> bool:
        """
        Returns:
            bool: whether the field called `name` exists.
        """
        return name in self._fields

    def remove(self, name: str) -> None:
        """
        Remove the field called `name`.
        """
        del self._fields[name]

    def clear(self) -> None:
        self._fields.clear()

    def get(self, name: str) -> Any:
        """
        Returns the field called `name`.
        """
        return self._fields[name]

    def get_fields(self) -> Dict[str, Any]:
        """
        Returns:
            dict: a dict which maps names (str) to data of the fields

        Modifying the returned dict will modify this instance.
        """
        return self._fields

    # Tensor-like methods
    def to(self, *args: Any, **kwargs: Any) -> "TrackletStates":
        """
        Returns:
            TrackletStates: all fields are called with a `to(device)`, if the field has this method.
        """
        ret = TrackletStates()
        for k, v in self._fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            ret.set(k, v)
        return ret

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "TrackletStates":
        """
        Args:
            item: an index-like object and will be used to index all the fields.

        Returns:
            If `item` is a string, return the data in the corresponding field.
            Otherwise, returns a `TrackletStates` where all fields are indexed by `item`.
        """
        if type(item) == int:
            if item >= len(self) or item < -len(self):
                raise IndexError("TrackletStates index out of range!")
            else:
                item = slice(item, None, len(self))

        ret = TrackletStates()
        for k, v in self._fields.items():
            ret.set(k, v[item])
        return ret

    def __len__(self) -> int:
        for v in self._fields.values():
            # use __len__ because len() has to be int and is not friendly to tracing
            return v.__len__()
        return 0
        # raise NotImplementedError("Empty TrackletStates does not support __len__!")

    def __iter__(self):
        raise NotImplementedError("`TrackletStates` object is not iterable!")

    @staticmethod
    def cat(tracklet_lists: List["TrackletStates"]) -> "TrackletStates":
        """
        Args:
            tracklet_lists (list[TrackletStates])

        Returns:
            TrackletStates
        """
        assert all(isinstance(i, TrackletStates) for i in tracklet_lists)
        assert len(tracklet_lists) > 0
        if len(tracklet_lists) == 1:
            return tracklet_lists[0]

        ret = TrackletStates()
        for k in tracklet_lists[0]._fields.keys():
            values = [i.get(k) for i in tracklet_lists]
            v0 = values[0]
            if isinstance(v0, torch.Tensor):
                values = torch.cat(values, dim=0)
            elif isinstance(v0, list):
                values = list(itertools.chain(*values))
            elif hasattr(type(v0), "cat"):
                values = type(v0).cat(values)
            else:
                raise ValueError("Unsupported type {} for concatenation".format(type(v0)))
            ret.set(k, values)
        return ret

    def __str__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_tracklets={}, ".format(len(self))
        s += "fields=[{}])".format(", ".join((f"{k}: {v}" for k, v in self._fields.items())))
        return s

    __repr__ = __str__
