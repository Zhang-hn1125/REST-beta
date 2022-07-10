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
mkfolder(settings.result_dir) 



for mrc in settings.mrc_list:
    
    root_name = mrc.split('/')[-1].split('.')[0]
    extension = mrc.split('/')[-1].split('.')[1]
    with mrcfile.open(mrc) as mrcData:
        orig_data = normalize(mrcData.data.astype(np.float32)*-1, percentile = settings.normalize_percentile)
#     orig_data = apply_wedge(orig_data, ld1=1, ld2=0)
    orig_data = normalize(orig_data, percentile = settings.normalize_percentile)

    with mrcfile.new('{}/{}_iter00.{}'.format(settings.result_dir,root_name, extension), overwrite=True) as output_mrc:
        output_mrc.set_data(-orig_data)

num_iter=1
logging.info("Start Iteration{}!".format(num_iter))
# pretrained_model case
if num_iter ==1:
    args = prepare_first_model(args)
else:
    args.init_model = '{}/model_iter{:0>2d}.h5'.format(args.result_dir,args.iter_count-1)

noise_level_series = get_noise_level(args.noise_level,args.noise_start_iter,args.iterations)
args.noise_level_current =  noise_level_series[num_iter]
dirs_tomake = ['train_x','train_y', 'test_x', 'test_y']
if not os.path.exists(settings.data_dir):
    os.makedirs(settings.data_dir)
for d in dirs_tomake:
    folder = '{}/{}'.format(settings.data_dir, d)
    if not os.path.exists(folder):
        os.makedirs(folder)
inp=[]
for i,mrc in enumerate(settings.mrc_list):
    inp.append((mrc, i*len(rotation_list)))
args.noise_level = (0.05,0.10,0.15,0.20)

args.noise_start_iter = (11,16,21,26)

args.noise_mode = 'noFilter'

args.noise_dir = args.result_dir +'/training_noise'

args.log_level = "info"

for k,i in enumerate(inp):
    mrc, start = i
   
    root_name = mrc.split('/')[-1].split('.')[0]
    current_mrc = '{}/{}_iter{:0>2d}.mrc'.format(settings.result_dir,root_name,0)
    print(current_mrc)
    with mrcfile.open(current_mrc) as mrcData:
        ow_data = mrcData.data.astype(np.float32)
    ow_data = normalize(ow_data, percentile = settings.normalize_percentile)
    with mrcfile.open('{}/{}_iter00.mrc'.format(settings.result_dir,root_name)) as mrcData:
        iw_data = mrcData.data.astype(np.float32)
    iw_data = normalize(iw_data, percentile = settings.normalize_percentile)


    orig_data = ow_data

    data=orig_data
    print('k',k)
    data_cubes = DataCubes(data, nCubesPerImg=1, cubeSideLen = settings.cube_size, cropsize = settings.crop_size, 
    mask = None, noise_folder = settings.noise_dir,noise_level = settings.noise_level_current,noise_mode = settings.noise_mode)
    print('start:',start)
    for i,img in enumerate(data_cubes.cubesX):
#         print('i+start',i+start)
        with mrcfile.new('{}/train_x/x_{}.mrc'.format(settings.data_dir, i+start), overwrite=True) as output_mrc:
            output_mrc.set_data(img.astype(np.float32))
        with mrcfile.new('{}/train_y/y_{}.mrc'.format(settings.data_dir, i+start), overwrite=True) as output_mrc:
            output_mrc.set_data(data_cubes.cubesY[i].astype(np.float32))
    start += 1#settings.ncube



