import os, glob

def delete():
    folder = 'form-input/'
    files = glob.glob(os.path.join(folder, "*"))
    for file in files:
        print(file)
        os.remove(file)
        print("old audio file has been deleted.")