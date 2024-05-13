import argparse
import numpy as np
from PIL import Image
import os
join = os.path.join
import stain_utils as utils
import stainNorm_Reinhard
import stainNorm_Macenko
import stainNorm_Vahadane
import numpy as np
import matplotlib.pyplot as plt
import time

OPENBLAS_NUM_THREADS=1
# export OPENBLAS_NUM_THREADS=1


if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--imageDir', type=str, default='/vast/AI_team/dataset/color_test/3dh/train/M', help='RGB image file')
    parser.add_argument('--saveDir', type=str, default='/vast/AI_team/dataset/color_test/StainNorm/Target_Leica/StainNorm_Vahadane/3dh/train/M', help='save file')
    args = parser.parse_args()
    
    input_list = sorted(next(os.walk(args.imageDir))[2])
    os.makedirs(args.saveDir, exist_ok=True)
    if "Thumbs.db" in input_list:
        input_list.remove("Thumbs.db")
    
    print('input len : ', len(input_list))

    # Setting Stain Norm methods
    # n=stainNorm_Reinhard.Normalizer()
    # n=stainNorm_Macenko.Normalizer()
    n=stainNorm_Vahadane.Normalizer()
    


    # Set Target Image
    # target_path = '/vast/AI_team/dataset/color_test/3dh/train/N/2018S016590902-22_31.jpg'   # 3D-HISTECH target
    target_path = '/vast/AI_team/dataset/color_test/leica/train/N/2022S 0326585050101 [d=4,x=8704,y=9728,w=1024,h=1024].jpg' # Leica target
    target_img = np.array(Image.open(target_path))
    n.fit(target_img)


    for case in input_list:
        t1 = time.time()
        img = np.array(Image.open(join(args.imageDir, case)))
        output_path = join(args.saveDir, case) 

        normalized = n.transform(img)
        
        # save results
        output_path = join(args.saveDir, case) 
        Image.fromarray(normalized).save(output_path)

        t2 = time.time()
        print(case, " costing :{0:03f}".format(t2-t1))

        