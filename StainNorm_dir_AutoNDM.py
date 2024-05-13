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
    parser.add_argument('--imageDir', type=str, default='/vast/AI_team/dataset/stomach_new/leica', help='RGB image file')
    parser.add_argument('--saveDir', type=str, default='/vast/AI_team/dataset/stomach/StainNorm/Target_3dh/StainNorm_Reinhard/leica', help='save file')
    args = parser.parse_args()
    
    input_list_dir = sorted(next(os.walk(args.imageDir))[1])
    os.makedirs(args.saveDir, exist_ok=True)

    # Setting Stain Norm methods
    n=stainNorm_Reinhard.Normalizer()
    # n=stainNorm_Macenko.Normalizer()
    # n=stainNorm_Vahadane.Normalizer()
    

    # Set Target Image
    target_path = '/vast/AI_team/dataset/color_test/3dh/train/N/2018S016590902-22_31.jpg'   # 3D-HISTECH target
    # target_path = '/vast/AI_team/dataset/color_test/leica/train/N/2022S 0326585050101 [d=4,x=8704,y=9728,w=1024,h=1024].jpg' # Leica target
    target_img = np.array(Image.open(target_path))
    n.fit(target_img)


    for t_list in input_list_dir:
        print(t_list)
        train_test = sorted(next(os.walk(join(args.imageDir, t_list)))[1])
        
        for NDM_list in train_test:
            print(NDM_list)
            NDM_path = join(args.imageDir, t_list, NDM_list)
            case_list = sorted(next(os.walk(NDM_path))[2])
            print('input len : ', len(case_list))

            for case in case_list:
                t1 = time.time()
                if "Thumbs.db" in case_list:
                    case_list.remove("Thumbs.db")

                img = np.array(Image.open(join(NDM_path, case)))
                output_path = join(args.saveDir, t_list, NDM_list, case)
                os.makedirs(join(args.saveDir, t_list, NDM_list), exist_ok=True) 

                normalized = n.transform(img)
                
                # save results
                Image.fromarray(normalized).save(output_path)

                t2 = time.time()
                print(case, " costing :{0:03f}".format(t2-t1))

        