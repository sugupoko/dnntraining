import cv2
import numpy as np
from glob import glob

num_classes = 2
img_height, img_width = 64, 64

CLS = ['akahara', 'madara']

def data_load(path, hf=False, vf=False, rot=None):
    xs = []
    ts = []
    paths = []
    
    for dir_path in glob(path + '/*'):
        for path in glob(dir_path + '/*'):
            print('x =',path)
            x = cv2.imread(path)
            x = cv2.resize(x, (img_width, img_height)).astype(np.float32)
            x /= 255.
            xs.append(x)

            for i, cls in enumerate(CLS):
                if cls in path:
                    print('t =', i)
                    t = i
            
            ts.append(t)

            paths.append(path)
            
            # 水平
            if hf:
                xs.append(x[:, ::-1])
                ts.append(t)
                paths.append(path)

            # 垂直
            if vf:
                xs.append(x[::-1])
                ts.append(t)
                paths.append(path)

            # 水平＆垂直
            if hf and vf:
                xs.append(x[::-1, ::-1])
                ts.append(t)
                paths.append(path)

            # 回転
            if rot is not None:
                angle = 0
                scale = 1
                while angle < 360:
                    angle += rot
                    _h, _w, _c = x.shape
                    max_side = max(_h, _w)
                    tmp = np.zeros((max_side, max_side, _c))
                    tx = int((max_side - _w) / 2)
                    ty = int((max_side - _h) / 2)
                    tmp[ty: ty+_h, tx: tx+_w] = x.copy()
                    M = cv2.getRotationMatrix2D((max_side/2, max_side/2), angle, scale)
                    _x = cv2.warpAffine(tmp, M, (max_side, max_side))
                    _x = _x[tx:tx+_w, ty:ty+_h]
                    xs.append(x)
                    ts.append(t)
                    paths.append(path)

    print(ts)
    xs = np.array(xs, dtype=np.float32)
    ts = np.array(ts, dtype=np.int)
    
    # 次元入れ変え (16, 64, 64, 3) -> (16, 3, 64, 64)
    xs = xs.transpose(0,3,1,2)

    return xs, ts, paths



if __name__ == '__main__':

    xs, ts, paths = data_load('./Dataset/train/images/', hf=True, vf=True, rot=30)
    print(xs.shape) # (16, 3, 64, 64)
    print(ts.shape) # (16,)

    mb = 3
    train_ind = np.arange(len(xs))
    print(train_ind) # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15]

    np.random.seed(0)
    np.random.shuffle(train_ind) 
    
    epoch_max = 3
    epoch = 0

    # 配列の頭から、3個づつ取り出す。
    while epoch < epoch_max:
        print('epoch:',epoch)
        mbi = 0

        for i in range(len(xs) // mb + 1):
            if mbi + mb > len(xs): # 例外処理残りの数が
                mb_ind = train_ind[mbi:]
                np.random.shuffle(train_ind)
                mb_ind = np.hstack((mb_ind, train_ind[:(mb-(len(xs)-mbi))]))
                mbi = mb - (len(xs) - mbi)
            else: # 基本的にこっち
                mb_ind = train_ind[mbi: mbi+mb]
                mbi += mb
            print(i, mb_ind)

        epoch += 1
        
            
   

    