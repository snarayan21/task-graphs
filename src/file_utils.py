from os import path


def clean_dir_name(dir_name, parent_dir):
    """Cleans the directory name within the parent directory such that there is no collision.
    If in the parent directory, the current directory name already exists, a '_#' is appended and
    incremented correctly.
    Note, this does not create the proposed directory, it merely cleans the name.
    Args:
        dir_name (str): Name of the proposed directory
        parent_dir (Path): Parent directory path
    Returns:
        proposed_dir (Path): Path to the clean proposed directory
        dir_name (str): Cleaned proposed directory name
    """
    proposed_dir = parent_dir / dir_name
    suffix_index = dir_name.rfind('_')
    underscore_exists = suffix_index != -1
    # Check if it ends in a number. If not, add one.
    if not (underscore_exists and dir_name[suffix_index + 1:].isdigit()):
        dir_name += '_0'
        proposed_dir = parent_dir / dir_name

    while path.isdir(proposed_dir):
        # Directory already exists. If it ends in _##, then we increment it
        suffix_index = dir_name.rfind('_')
        new_test_number = int(dir_name[suffix_index + 1:]) + 1
        dir_name = dir_name[:suffix_index + 1] + str(new_test_number)
        proposed_dir = parent_dir / dir_name
    return proposed_dir, dir_name


def clean_file_name(file_name, parent_dir):
    """Cleans the file name within the parent file such that there is no collision.
    If in the parent file, the current file name already exists, a '_#' is appended and
    incremented correctly.
    Note, this does not create the proposed file, it merely cleans the name.
    Args:
        file_name (str): Name of the proposed file (e.g. "model.zip")
        parent_dir (Path): Parent dir path (e.g. "/home/whatever")
    Returns:
        proposed_file (Path): Path to the clean proposed file (e.g. '/home/whatever/model_3.zip')
        file_name (str): Cleaned proposed file name (e.g. model_3.zip)
    """
    proposed_file = parent_dir / file_name
    ext = proposed_file.suffix  # extension, like .zip
    file_name_no_ext = proposed_file.stem  # just file_name without ext

    suffix_index = file_name_no_ext.rfind('_')
    underscore_exists = suffix_index != -1
    # Check if it ends in a number. If not, add one.
    if not (underscore_exists and file_name_no_ext[suffix_index + 1:].isdigit()):
        file_name_no_ext += '_0'
        proposed_file = proposed_file.with_name(file_name_no_ext).with_suffix(ext)

    while path.isfile(proposed_file):
        # File already exists. If it ends in _##, then we increment it
        suffix_index = file_name_no_ext.rfind('_')
        new_test_number = int(file_name_no_ext[suffix_index + 1:]) + 1
        file_name_no_ext = file_name_no_ext[:suffix_index + 1] + str(new_test_number)
        proposed_file = proposed_file.with_name(file_name_no_ext).with_suffix(ext)
    return proposed_file, proposed_file.name