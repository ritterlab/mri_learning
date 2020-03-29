def rename_file(path, old_name, new_name, old_ext, new_ext, extra_suffix=''):
    path = path.split('.')
    path[-2] = path[(-2)].replace(old_name, new_name) + extra_suffix
    path[-1] = path[(-1)].replace(old_ext, new_ext)
    return '.'.join(path)


def isinstance_none(obj, data_type):
    return isinstance(obj, data_type) or obj is None


def specificity(y_true, y_pred):
    return sum((y_true == 0) & (y_pred == y_true)) / sum(y_true == 0)