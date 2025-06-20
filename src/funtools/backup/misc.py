

def pop_dict_key(key, kwargs):

    if key not in kwargs: return False, None
    val = kwargs[key]
    del kwargs[key]
    return True, val


def pop_dict_keys(keys, kwargs):
    args = [(k,) + pop_dict_key(k, kwargs) for k in keys]
    return dict(**{k:d for k, is_found, d in args if is_found})