from os.path import join


def get_folders(direct):
    from os import listdir
    from os.path import isdir, join

    return [folder for folder in listdir(direct) if isdir(join(direct, folder))]

def get_files(folder, format=''):
    from os import listdir
    from os.path import isfile, join
    import re

    if format == '':
        return [file for file in listdir(folder) if isfile(join(folder, file))]
    else:
        return [file for file in listdir(folder)
                if isfile(join(folder, file)) & re.match(format, file)]