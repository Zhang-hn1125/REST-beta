
"""
Description:
  This is scripts is used to move file from a dir to the other dir
"""
import os,shutil
def movefile(oripath,tardir):
    filename = os.path.basename(oripath)
    tarpath = os.path.join(tardir, filename)
    if not os.path.exists(oripath):
        print('the dir is not exist:%s' % oripath)
        status = 0
    else:
        if os.path.exists(tardir):
            if os.path.exists(tarpath):
                os.remove(tarpath)
        else:
            os.makedirs(tardir)
        shutil.move(oripath, tardir)
        status = 1
    return status

movefile("process2results\data\train_y","results\data\train_y")

