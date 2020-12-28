import time
import numpy as np 
import tensorflow as tf 
from src.utils import get_batch, get_iteration, transform_y
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score,f1_score,cohen_kappa_score
from tqdm import tqdm

def predict_by_batch (model,test_S1,test_S2,test_MS,test_Pan,test_y,batch_size,lst_sensor,tqdm_disable): 
    '''
    Predict batch of test set
    '''
    pred=[]

    iteration = get_iteration(test_y,batch_size)
    if not tqdm_disable:
        print (f'Test batchs: {iteration}')

    start = time.time()
    if len(lst_sensor) == 3 :
        for batch in tqdm(range(iteration),disable=not(tqdm_disable)):
            batch_s1 = get_batch (test_S1,batch,batch_size)
            batch_s2 = get_batch (test_S2,batch,batch_size)
            batch_ms = get_batch (test_MS,batch,batch_size)
            batch_pan = get_batch (test_Pan,batch,batch_size)
            _,_,_,batch_pred = model(batch_s1,batch_s2,batch_ms,batch_pan,is_training=False)
            del batch_s1,batch_s2,batch_ms,batch_pan
            pred.append(tf.argmax(batch_pred,axis=1))

    elif len(lst_sensor) == 2 and 's1' in lst_sensor and 's2' in lst_sensor :
        for batch in tqdm(range(iteration),disable=not(tqdm_disable)):
            batch_s1 = get_batch (test_S1,batch,batch_size)
            batch_s2 = get_batch (test_S2,batch,batch_size)
            _,_,batch_pred = model(batch_s1,batch_s2,is_training=False)
            del batch_s1,batch_s2
            pred.append(tf.argmax(batch_pred,axis=1))
    
    elif len(lst_sensor) == 2 and 's2' in lst_sensor and 'spot' in lst_sensor :
        for batch in tqdm(range(iteration),disable=not(tqdm_disable)):
            batch_s2 = get_batch (test_S2,batch,batch_size)
            batch_ms = get_batch (test_MS,batch,batch_size)
            batch_pan = get_batch (test_Pan,batch,batch_size)
            _,_,batch_pred = model(batch_s2,batch_ms,batch_pan,is_training=False)
            del batch_s2, batch_ms, batch_pan
            pred.append(tf.argmax(batch_pred,axis=1))
    
    elif len(lst_sensor) == 1 and 's1' in lst_sensor :
        for batch in tqdm(range(iteration),disable=not(tqdm_disable)):
            batch_s1 = get_batch (test_S1,batch,batch_size)
            batch_pred = model(batch_s1,is_training=False)
            del batch_s1
            pred.append(tf.argmax(batch_pred,axis=1))
    
    elif len(lst_sensor) == 1 and 's2' in lst_sensor :
        for batch in tqdm(range(iteration),disable=not(tqdm_disable)):
            batch_s2 = get_batch (test_S2,batch,batch_size)
            batch_pred = model(batch_s2,is_training=False)
            del batch_s2
            pred.append(tf.argmax(batch_pred,axis=1))
    
    elif len(lst_sensor) == 1 and 'spot' in lst_sensor :
        for batch in tqdm(range(iteration),disable=not(tqdm_disable)):
            batch_ms = get_batch (test_MS,batch,batch_size)
            batch_pan = get_batch (test_Pan,batch,batch_size)
            batch_pred = model(batch_ms,batch_pan,is_training=False)
            del batch_ms, batch_pan
            pred.append(tf.argmax(batch_pred,axis=1))
    
    stop = time.time()
    elapsed = stop - start

    pred = np.hstack(pred)
    return pred, elapsed
        
def restore (model,test_S1,test_S2,test_MS,test_Pan,test_y,batch_size,checkpoint_path,result_path,lst_sensor,tqdm_disable):
    '''
    Load weights for best configuration and evaluate on test set
    '''
    model.load_weights(checkpoint_path)
    print ('Weights loaded')

    pred, elapsed = predict_by_batch (model,test_S1,test_S2,test_MS,test_Pan,test_y,batch_size,lst_sensor,tqdm_disable)
    print (f'Test Time: {elapsed}')
    pred = transform_y (test_y,pred)
    np.save (result_path,pred)

    print ('Acc:',accuracy_score(test_y,pred))
    print ('F1:',f1_score(test_y,pred,average='weighted'))
    print ('Kappa:',cohen_kappa_score(test_y,pred))