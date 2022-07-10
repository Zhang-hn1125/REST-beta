import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
from tensorflow.keras.models import load_model
import mrcfile
from REST.preprocessing.img_processing import normalize
import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf

def predict(settings):    
    # model = load_model('{}/model_iter{:0>2d}.h5'.format(settings.result_dir,settings.iter_count+1))
    strategy = tf.distribute.MirroredStrategy()
    if settings.ngpus >1:
        with strategy.scope():
            model = load_model('{}/model_iter{:0>2d}.h5'.format(settings.result_dir,settings.iter_count))
    else:
        model = load_model('{}/model_iter{:0>2d}.h5'.format(settings.result_dir,settings.iter_count))
    N = settings.predict_batch_size 
    num_batches = len(settings.mrc_list)
    if num_batches%N == 0:
        append_number = 0
    else:
        append_number = N - num_batches%N
    data = []
    for i,mrc in enumerate(settings.mrc_list + settings.mrc_list[:append_number]):
        root_name = mrc.split('/')[-1].split('.')[0]
        with mrcfile.open('{}/{}_iter00.mrc'.format(settings.result_dir,root_name)) as mrcData:
            real_data = mrcData.data.astype(np.float32)*-1
        real_data=normalize(real_data, percentile = settings.normalize_percentile)

        cube_size = real_data.shape[0]
        pad_size1 = (settings.predict_cropsize - cube_size)//2
        pad_size2 = pad_size1+1 if (settings.predict_cropsize - cube_size)%2 !=0 else  pad_size1
        padi = (pad_size1,pad_size2)
        real_data = np.pad(real_data, (padi,padi,padi), 'symmetric')

        if (i+1)%N != 0:
            data.append(real_data)
        else:
            data.append(real_data)
            data = np.array(data)
            predicted=model.predict(data[:,:,:,:,np.newaxis], batch_size= settings.predict_batch_size,verbose=0)
            predicted = predicted.reshape(predicted.shape[0:-1])
            for j,outData in enumerate(predicted):
                count = i + j - N + 1
                if count < len(settings.mrc_list):
                    m_name = settings.mrc_list[count]

                    root_name = m_name.split('/')[-1].split('.')[0]
                    end_size = pad_size1+cube_size
                    outData1 = outData[pad_size1:end_size, pad_size1:end_size, pad_size1:end_size]
                    outData1 = normalize(outData1, percentile = settings.normalize_percentile)
                    with mrcfile.new('{}/{}_iter{:0>2d}.mrc'.format(settings.result_dir,root_name,settings.iter_count), overwrite=True) as output_mrc:
                        output_mrc.set_data(-outData1)
            data = []
    K.clear_session()
    
