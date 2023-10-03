import os
import shutil


def create_directory(dirname: str, parent_dir: str) -> str:
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    full_path = os.path.join(parent_dir, dirname)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    return full_path


def delete_directory(dirpath: str):
    if not os.path.isdir(dirpath):
        raise NotADirectoryError(f"Path is not a directory: {dirpath}")
    shutil.rmtree(dirpath)


def split_path(full_path: str):
    """ Splits a full path into its components: path, filename and extension """
    path, extension = os.path.splitext(full_path)
    path, filename = os.path.split(path)
    return path, filename, extension
