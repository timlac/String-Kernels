#!/usr/bin/env python
from collections import Iterable
from random import shuffle
import numpy as np


class TwoWayDict(dict):
    def __setitem__(self, key, value):
        # Remove any previous connections with these values
        if key in self:
            del self[key]
        if value in self:
            del self[value]
        dict.__setitem__(self, key, value)
        dict.__setitem__(self, value, key)

    def __delitem__(self, key):
        dict.__delitem__(self, self[key])
        dict.__delitem__(self, key)

    def __len__(self):
        """Returns the number of connections"""
        return dict.__len__(self) // 2


def flatten(l):
    """
    Flatten a list of lists.

    Parameters
    ----------
    l : list

    Returns
    -------
    generator
        A generator containing all the elements of l and its sublists.

    """
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


def shuffle_lists(*lists):
    """
    Shuffle N lists.

    The lists are shuffled in the exact same way.

    Parameters
    ----------
    *lists
        One or more lists

    Returns
    -------
    *lists
        Returns the N lists, shuffled.

    """
    l = list(zip(*lists))
    shuffle(l)
    return zip(*l)


def filter_dict(key_list_filter, dictionary):
    """
    Filters a dict by a list of keys.

    This function makes sures only keys given by a list are in the dict.

    Parameters
    ----------
    key_list_filter : list
        The keys that should remain in the dict.
    dictionary : dict
        The dictionary to be filtered.

    Returns
    -------
    dict
        The filtered dict.

    """
    return {key: dictionary[key] for key in key_list_filter}


def prepend_string_to_array(text, array_or_list):
    """
    Prepends a string to a numpy array or a list.

    Parameters
    ----------
    text : str
        The string to be prepended.
    array_or_list : list or numpy.ndarray
        The numpy array.

    Returns
    -------
    list
        Returns the new list.

    """
    if isinstance(array_or_list, np.ndarray):
            array_or_list = array_or_list.tolist()
    array_or_list.insert(0, text)
    return array_or_list
