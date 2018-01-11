#!/usr/bin/env python


def split_data(index_filter, index):
    return {key: index[key] for key in index_filter}
