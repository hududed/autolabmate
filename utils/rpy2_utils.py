from typing import Dict, Any
import rpy2.robjects as ro


def py_dict_to_r_list(py_dict: Dict[str, Any]) -> ro.ListVector:
    r_list = ro.ListVector({})
    for k, v in py_dict.items():
        if isinstance(v, dict):
            r_list.rx2[k] = py_dict_to_r_list(v)
        elif isinstance(v, list):
            r_list.rx2[k] = ro.StrVector([str(i) for i in v])
        else:
            r_list.rx2[k] = ro.StrVector([str(v)])
    return r_list


def py_dict_to_r_named_vector(py_dict) -> ro.StrVector:
    r_vector = ro.StrVector([])
    names = []
    for k, v in py_dict.items():
        if isinstance(v, dict):
            r_vector = ro.r.c(r_vector, py_dict_to_r_named_vector(v))
            names.extend([f"{k}.{sub_k}" for sub_k in v.keys()])
        elif isinstance(v, list):
            r_vector = ro.r.c(r_vector, ro.StrVector([str(i) for i in v]))
            names.extend([k] * len(v))
        else:
            r_vector = ro.r.c(r_vector, ro.StrVector([str(v)]))
            names.append(k)
    r_vector.names = ro.StrVector(names)
    return r_vector
