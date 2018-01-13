#!/usr/bin/env python
from random import shuffle


def shuffle_lists(*lists):
    """
    Shuffle an arbitrary number of lists.

    Parameters
    ----------
    *lists
        One or more lists

    Returns
    -------
    *lists
        Returns the same number of list as the input, shuffled.

    """
    l = list(zip(*lists))
    shuffle(l)
    return zip(*l)


def filter_dict(index_filter, index):
    return {key: index[key] for key in index_filter}


def preppend_string_to_array(text, array_or_list):
    if isinstance(array_or_list, np.ndarray):
            array_or_list = array_or_list.tolist()
    array_or_list.insert(0, text)
    return array_or_list
