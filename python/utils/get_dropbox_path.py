import os
dropbox_folder = None

for dirname, dirnames, filenames in os.walk(os.path.expanduser('~')):
    for subdirname in dirnames:
        if(subdirname == 'Dropbox'):
            dropbox_folder = os.path.join(dirname, subdirname)
            break
    if dropbox_folder:
        break

print(dropbox_folder)
