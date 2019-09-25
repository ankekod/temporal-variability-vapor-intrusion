import os
dropbox_folder = None

def get_dropbox_path():
    for dirname, dirnames, filenames in os.walk(os.path.expanduser('~')):
        for subdirname in dirnames:
            if(subdirname == 'Dropbox'):
                dropbox_folder = os.path.join(dirname, subdirname)
                break
        if dropbox_folder:
            break
    return dropbox_folder


def get_epa_house_db_path(path='/var/Indianapolis.db'):
    return get_dropbox_path()+path


def get_asu_house_db_path(path='/var/HillAFB.db'):
    return get_dropbox_path()+path

print(get_asu_house_db_path())
