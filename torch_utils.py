import os

FOLDERS = ['images', 'network', 'numerical', 'plots']


def make_dir_tree(parent_dir):
    for folder in FOLDERS:
        new_dir = '{}/{}'.format(parent_dir, folder)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)






