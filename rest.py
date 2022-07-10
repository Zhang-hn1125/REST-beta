
import fire
import logging
import os, sys, traceback
from REST.util.dict2attr import Arg,check_parse,idx2list
from fire import core
from REST.util.metadata import MetaData,Label,Item

class REST:
    """
    REST: Train on tomograms and Predict to restore missing-wedge\n
    for detail discription, run one of the following commands:

    REST.py prepare_star -h
    REST.py prepare_subtomo_star -h
    REST.py deconv -h
    REST.py make_mask -h
    REST.py extract -h
    REST.py refine -h
    REST.py predict -h

    """
    #log_file = "log.txt"


    def predict(self, star_file: str, model: str, output_dir: str='./corrected_tomos', gpuID: str = None, cube_size:int=48,
    crop_size:int=64,use_deconv_tomo=True, batch_size:int=None,normalize_percentile: bool=True,log_level: str="info", tomo_idx=None):
        """
        \nPredict tomograms using trained model\n
        REST.py predict star_file model [--gpuID] [--output_dir] [--cube_size] [--crop_size] [--batch_size] [--tomo_idx]
        :param star_file: star for tomograms.
        :param output_dir: file_name of output predicted tomograms
        :param model: path to trained network model .h5
        :param gpuID: (0,1,2,3) The gpuID to used during the training. e.g 0,1,2,3.
        :param cube_size: (64) The tomogram is divided into cubes to predict due to the memory limitation of GPUs.
        :param crop_size: (96) The side-length of cubes cropping from tomogram in an overlapping patch strategy, make this value larger if you see the patchy artifacts
        :param batch_size: The batch size of the cubes grouped into for network predicting
        :param normalize_percentile: (True) if normalize the tomograms by percentile. Should be the same with that in refine parameter.
        :param log_level: ("debug") level of message to be displayed, could be 'info' or 'debug'
        :param tomo_idx: (None) If this value is set, process only the tomograms listed in this index. e.g. 1,2,4 or 5-10,15,16
        :param use_deconv_tomo: (True) If CTF deconvolved tomogram is found in tomogram.star, use that tomogram instead.
        :raises: AttributeError, KeyError
        """
        d = locals()
        d_args = Arg(d)
        from REST.bin.predict import predict


        if d_args.log_level == "debug":
            logging.basicConfig(format='%(asctime)s, %(levelname)-8s %(message)s',
            datefmt="%m-%d %H:%M:%S",level=logging.DEBUG,handlers=[logging.StreamHandler(sys.stdout)])
        else:
            logging.basicConfig(format='%(asctime)s, %(levelname)-8s %(message)s',
            datefmt="%m-%d %H:%M:%S",level=logging.INFO,handlers=[logging.StreamHandler(sys.stdout)])
        try:
            predict(d_args)
        except:
            error_text = traceback.format_exc()
            f =open('log.txt','a+')
            f.write(error_text)
            f.close()
            logging.error(error_text)

    def check(self):
        from REST.bin.predict import predict
        from REST.bin.refine import run
        import skimage
        import PyQt5
        import tqdm
        print('REST --version 0.1 installed')

    def gui(self):
        import REST.gui.REST_star_app as app
        app.main()

def Display(lines, out):
    text = "\n".join(lines) + "\n"
    out.write(text)

def pool_process(p_func,chunks_list,ncpu):
    from multiprocessing import Pool
    with Pool(ncpu,maxtasksperchild=1000) as p:
        # results = p.map(partial_func,chunks_gpu_num_list,chunksize=1)
        results = list(p.map(p_func,chunks_list))
    # return results

if __name__ == "__main__":
    core.Display = Display
    # logging.basicConfig(format='%(asctime)s, %(levelname)-8s %(message)s',datefmt="%m-%d %H:%M:%S",level=logging.INFO)
    if len(sys.argv) > 1:
        check_parse(sys.argv[1:])
    fire.Fire(REST)
