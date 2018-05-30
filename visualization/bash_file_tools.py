import os

def sort_string_with_numbers(str_list):
    def number_str_comparator(a,b):
        import re
        a_num = float(re.search(r'\d+.\d+|\d+',a).group())
        b_num = float(re.search(r'\d+.\d+|\d+',b).group())
        return a_num - b_num

    import functools
    return sorted(str_list, key=functools.cmp_to_key(number_str_comparator))

def list_bash_files(model_dirs_bash_path):
    import re
    cmd = "echo " + model_dirs_bash_path
    model_dirs = os.popen(cmd).read()
    model_dirs = model_dirs.split()
    model_dirs = sort_string_with_numbers(model_dirs)
    return model_dirs

