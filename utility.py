import os

def allFilePath(data_dir):
    file_list = []
    for (dirpath, dirnames, filenames) in os.walk(data_dir):
        for filename in filenames:
            file_dir = os.path.join(dirpath, filename)
            file_list.append(file_dir)
    return file_list
