import logging,glob,mrcfile,os,sys,shutil,traceback
from REST.preprocessing.cubes import prepare_cubes
from REST.preprocessing.img_processing import normalize
from REST.preprocessing.prepare import prepare_first_iter,get_cubes_list,get_noise_level
from REST.util.dict2attr import save_args_json,load_args_from_json
import numpy as np
from REST.util.metadata import MetaData, Item, Label
from REST.util.utils import mkfolder
from REST.util.dict2attr import Arg,check_parse,idx2list
from REST.util.dict2attr import Arg,check_parse,idx2list
from REST.util.metadata import MetaData,Label,Item
from REST.preprocessing.cubes import create_cube_seeds,crop_cubes,DataCubes
from REST.preprocessing.img_processing import normalize
from REST.preprocessing.simulate import apply_wedge1 as  apply_wedge
from multiprocessing import Pool
from functools import partial
from REST.util.rotations import rotation_list
from REST.training.predict import predict
from REST.training.train import prepare_first_model, train_data
from REST.preprocessing.cubes import create_cube_seeds,crop_cubes,DataCubes
from REST.preprocessing.img_processing import normalize
from REST.preprocessing.simulate import apply_wedge1 as  apply_wedge
from multiprocessing import Pool
from functools import partial
from REST.training.train import prepare_first_model, train_data,train3D_continue

dic={ 'gpuID': None,
 'iterations': None,
 'data_dir': None,
 'pretrained_model': None,
 'log_level': None,
 'result_dir': 'results',
 'preprocessing_ncpus': 16,
 'continue_from': None,
 'epochs': 10,
 'batch_size': None,
 'steps_per_epoch': None,
 'noise_level': None,
 'noise_start_iter': None,
 'noise_mode': None,
 'noise_dir': None,
 'learning_rate': None,
 'drop_out': 0.3,
 'convs_per_depth': 3,
 'kernel': (3, 3, 3),
 'pool': None,
 'unet_depth': 3,
 'filter_base': None,
 'batch_normalization': False,
 'normalize_percentile': True,
'subtomo_star':'subtomo.star'}

args=Arg(dic)
md = MetaData()
md.read(args.subtomo_star)
# args.crop_size = 18
# args.cube_size = 18
# args.predict_cropsize = 18
# num_noise_volume = 1000
# args.residual = False
args.crop_size = md._data[0].rlnCropSize
args.cube_size = md._data[0].rlnCubeSize
args.predict_cropsize = args.crop_size
num_noise_volume = 3000
args.residual = False
args.gpuID = 0
if args.data_dir is None:
    args.data_dir = args.result_dir + '/data'
if args.iterations is None:
    args.iterations = 30
args.ngpus = 1
if args.result_dir is None:
    args.result_dir = 'results'
if args.batch_size is None:
    args.batch_size = max(4, 2 * args.ngpus)
args.predict_batch_size = args.batch_size
if args.filter_base is None:
    if md._data[0].rlnPixelSize >15:
        args.filter_base = 32
    else:
        args.filter_base = 64
if args.steps_per_epoch is None:
    args.steps_per_epoch = min(int(len(md) * 6/args.batch_size) , 200)
if args.learning_rate is None:
    args.learning_rate = 0.0004
if args.noise_level is None:
    args.noise_level = (0.05,0.10,0.15,0.20)
if args.noise_start_iter is None:
    args.noise_start_iter = (11,16,21,26)
if args.noise_mode is None:
    args.noise_mode = 'noFilter'
if args.noise_dir is None:
    args.noise_dir = args.result_dir +'/training_noise'
if args.log_level is None:
    args.log_level = "info"

if len(md) <=0:
    logging.error("Subtomo list is empty!")
    sys.exit(0)
args.mrc_list = []


for i,it in enumerate(md):
    if "rlnImageName" in md.getLabels():
        args.mrc_list.append(it.rlnImageName)


settings=args
num_iter=1
logging.info("Start Iteration{}!".format(num_iter))
# pretrained_model case
if num_iter ==1:
    args = prepare_first_model(args)
else:
    args.init_model = '{}/model_iter{:0>2d}.h5'.format(args.result_dir,args.iter_count-1)

noise_level_series = get_noise_level(args.noise_level,args.noise_start_iter,args.iterations)
args.noise_level_current =  noise_level_series[num_iter]
args.noise_level = (0.05,0.10,0.15,0.20)

args.noise_start_iter = (11,16,21,26)

args.noise_mode = 'noFilter'

args.noise_dir = args.result_dir +'/training_noise'

args.log_level = "info"
all_path_x = os.listdir(settings.data_dir+'/train_x')
num_test = int(len(all_path_x) * 0.1) 
num_test = num_test - num_test%settings.ngpus + settings.ngpus
all_path_y = ['y_'+i.split('_')[1] for i in all_path_x ]
ind = np.random.permutation(len(all_path_x))[0:num_test]
for i in ind:
    os.rename('{}/train_x/{}'.format(settings.data_dir, all_path_x[i]), '{}/test_x/{}'.format(settings.data_dir, all_path_x[i]) )
    os.rename('{}/train_y/{}'.format(settings.data_dir, all_path_y[i]), '{}/test_y/{}'.format(settings.data_dir, all_path_y[i]) )
    #os.rename('data/train_y/'+all_path_y[i], 'data/test_y/'+all_path_y[i])
history = train3D_continue('{}/model_iter{:0>2d}.h5'.format(settings.result_dir,0),
                                        settings.init_model,
                                        data_dir = settings.data_dir,
                                        result_folder = settings.result_dir,
                                        epochs=2,
                                        steps_per_epoch=1500,
                                        batch_size=settings.batch_size,
                                        lr = 0.0001,
                                        n_gpus=1)

