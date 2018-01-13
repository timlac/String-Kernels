#!/usr/bin/env python


def filter_dict(index_filter, index):
    return {key: index[key] for key in index_filter}
