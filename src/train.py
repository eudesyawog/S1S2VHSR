import time
import numpy as np
import os
import pandas as pd
import tensorflow as tf 
from src.utils import get_batch, get_iteration
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from tqdm import tqdm

def save_history (dic, epoch, loss, acc, partition):
    '''
    Add training information to a dictionary
    '''
    dic.setdefault('Epoch',[]).append(epoch)
    dic.setdefault('Loss',[]).append(loss)
    dic.setdefault('Accuracy',[]).append(acc)
    dic.setdefault('Partition',[]).append(partition)

def train_info (model,checkpoint_path,epoch,train_loss,train_acc,valid_loss,valid_acc,elapsed,best_acc,valid_y,pred):
    '''
    Output of training step
    Save model if accuracy improves
    '''
    print (f'Epoch {epoch+1}, Loss: {train_loss.result().numpy()}, Acc: {train_acc.result().numpy()}, Valid Loss: {valid_loss.result().numpy()}, Valid Acc: {valid_acc.result().numpy()}, Time: {elapsed}')
    if valid_acc.result().numpy() > best_acc :
        print ( f1_score (valid_y,pred,average=None) )
        model.save_weights(checkpoint_path)
        print (f'{valid_acc.name} improved from {best_acc} to {valid_acc.result().numpy()}, saving to {checkpoint_path}')
        best_acc = valid_acc.result().numpy()
        
    # Reset metrics for the next epoch
    train_loss.reset_states()
    train_acc.reset_states()
    valid_loss.reset_states()
    valid_acc.reset_states()

    if epoch == 0:
        print (model.summary())

    return best_acc

@tf.function
def train_step (model, x_s1, x_s2, x_ms, x_pan, y, loss_fn, distill_loss_fn, optimizer, loss, metric, lst_sensor, weight, supervision, is_training):
    '''
    Gradient differentiation
    '''
    with tf.GradientTape() as tape:
        if len (lst_sensor) == 3 :
            s1_pred, s2_pred, spot_pred, main_pred = model(x_s1, x_s2, x_ms, x_pan,is_training)
            cost = loss_fn(y,main_pred)
            if not supervision is None :
                if supervision == 'distill':
                    cost_s1 = distill_loss_fn(main_pred,s1_pred)
                    cost_s2 = distill_loss_fn(main_pred,s2_pred)
                    cost_spot = distill_loss_fn(main_pred,spot_pred)
                elif supervision == 'labels':
                    cost_s1 = loss_fn(y,s1_pred)
                    cost_s2 = loss_fn(y,s2_pred)
                    cost_spot = loss_fn(y,spot_pred)
                cost+= weight*cost_s1 + weight*cost_s2 + weight*cost_spot
        elif len (lst_sensor) == 2  and 's1' in lst_sensor and 's2' in lst_sensor :
            s1_pred, s2_pred, main_pred = model(x_s1, x_s2, is_training)
            cost = loss_fn(y,main_pred)
            if not supervision is None :
                if supervision == 'distill':
                    cost_s1 = distill_loss_fn(main_pred,s1_pred)
                    cost_s2 = distill_loss_fn(main_pred,s2_pred)
                elif supervision == 'labels':
                    cost_s1 = loss_fn(y,s1_pred)
                    cost_s2 = loss_fn(y,s2_pred)
                cost+= weight*cost_s1 + weight*cost_s2
        elif len (lst_sensor) == 2  and 's2' in lst_sensor and 'spot' in lst_sensor :
            s2_pred, spot_pred, main_pred = model(x_s2, x_ms, x_pan, is_training)
            cost = loss_fn(y,main_pred)
            if not supervision is None :
                if supervision == 'distill':
                    cost_s2 = distill_loss_fn(main_pred,s2_pred)
                    cost_spot = distill_loss_fn(main_pred,spot_pred)
                elif supervision == 'labels':
                    cost_s2 = loss_fn(y,s2_pred)
                    cost_spot = loss_fn(y,spot_pred)
                cost+= weight*cost_s2 + weight*cost_spot
        elif len (lst_sensor) == 1  and 's1' in lst_sensor :
            main_pred = model(x_s1, is_training)
            cost = loss_fn(y,main_pred)
        elif len (lst_sensor) == 1  and 's2' in lst_sensor :
            main_pred = model(x_s2, is_training)
            cost = loss_fn(y,main_pred)
        elif len (lst_sensor) == 1  and 'spot' in lst_sensor :
            main_pred = model(x_ms, x_pan, is_training)
            cost = loss_fn(y,main_pred)

        if is_training :
            gradients = tape.gradient(cost, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        loss(cost)
        metric(y, tf.math.argmax(main_pred,axis=1))
    return  tf.math.argmax(main_pred,axis=1)

def run (model,train_S1,train_S2,train_MS,train_Pan,train_y,
            valid_S1,valid_S2,valid_MS,valid_Pan,valid_y,
                checkpoint_path,batch_size,lr,n_epochs,
                    lst_sensor,weight,supervision,train_hist,tqdm_display) :
    '''
    Main function for training models
    '''
    
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    distill_loss_fn = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate = lr)
    
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_acc = tf.keras.metrics.Accuracy(name='train_acc')
    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_acc = tf.keras.metrics.Accuracy(name='valid_acc')
    
    best_acc = float("-inf")

    train_iter = get_iteration (train_y,batch_size)
    valid_iter = get_iteration (valid_y,batch_size)
    if not tqdm_display:
        print (f'Training batchs: {train_iter}')
        print (f'Validation batchs: {valid_iter}')
    
    history_dic = {}
   
    if len (lst_sensor) == 3:
        for epoch in range(n_epochs):
            start = time.time()
            train_S1, train_S2, train_MS, train_Pan, train_y = shuffle(train_S1, train_S2, train_MS, train_Pan, train_y, random_state=0)
            for batch in tqdm(range(train_iter),disable=not(tqdm_display)):
                batch_s1 = get_batch (train_S1,batch,batch_size)
                batch_s2 = get_batch (train_S2,batch,batch_size)
                batch_ms = get_batch (train_MS,batch,batch_size)
                batch_pan = get_batch (train_Pan,batch,batch_size)
                batch_y = get_batch (train_y,batch,batch_size)
                train_step(model,batch_s1,batch_s2,batch_ms,batch_pan,batch_y,loss_fn,distill_loss_fn,optimizer,train_loss,train_acc,lst_sensor,weight,supervision,is_training=True)
                del batch_s1,batch_s2,batch_ms,batch_pan,batch_y
            pred = []
            for batch in tqdm(range(valid_iter),disable=not(tqdm_display)):
                batch_s1 = get_batch (valid_S1,batch,batch_size)
                batch_s2 = get_batch (valid_S2,batch,batch_size)
                batch_ms = get_batch (valid_MS,batch,batch_size)
                batch_pan = get_batch (valid_Pan,batch,batch_size)
                batch_y = get_batch (valid_y,batch,batch_size)
                batch_pred = train_step(model,batch_s1,batch_s2,batch_ms,batch_pan,batch_y,loss_fn,distill_loss_fn,optimizer,valid_loss,valid_acc,lst_sensor,weight,supervision,is_training=False)
                del batch_s1,batch_s2,batch_ms,batch_pan,batch_y
                pred.append(batch_pred)
            pred = np.hstack(pred)
            stop = time.time()
            elapsed = stop - start
            if train_hist:
                save_history (history_dic, epoch+1, train_loss.result().numpy(), train_acc.result().numpy(), 'Training')
                save_history (history_dic, epoch+1, valid_loss.result().numpy(), valid_acc.result().numpy(), 'Validation')
            best_acc = train_info (model,checkpoint_path,epoch,train_loss,train_acc,valid_loss,valid_acc,elapsed,best_acc,valid_y,pred)

    elif len (lst_sensor) == 2  and 's1' in lst_sensor and 's2' in lst_sensor:
        for epoch in range(n_epochs):
            start = time.time()
            train_S1, train_S2, train_y = shuffle(train_S1, train_S2, train_y, random_state=0)
            for batch in tqdm(range(train_iter),disable=not(tqdm_display)):
                batch_s1 = get_batch (train_S1,batch,batch_size)
                batch_s2 = get_batch (train_S2,batch,batch_size)
                batch_y = get_batch (train_y,batch,batch_size)
                train_step(model,batch_s1,batch_s2,None,None,batch_y,loss_fn,distill_loss_fn,optimizer,train_loss,train_acc,lst_sensor,weight,supervision,is_training=True)
                del batch_s1,batch_s2,batch_y
            pred = []
            for batch in tqdm(range(valid_iter),disable=not(tqdm_display)):
                batch_s1 = get_batch (valid_S1,batch,batch_size)
                batch_s2 = get_batch (valid_S2,batch,batch_size)
                batch_y = get_batch (valid_y,batch,batch_size)
                batch_pred = train_step(model,batch_s1,batch_s2,None,None,batch_y,loss_fn,distill_loss_fn,optimizer,valid_loss,valid_acc,lst_sensor,weight,supervision,is_training=False)
                del batch_s1,batch_s2,batch_y
                pred.append(batch_pred)
            pred = np.hstack(pred)
            stop = time.time()
            elapsed = stop - start
            if train_hist:
                save_history (history_dic, epoch+1, train_loss.result().numpy(), train_acc.result().numpy(), 'Training')
                save_history (history_dic, epoch+1, valid_loss.result().numpy(), valid_acc.result().numpy(), 'Validation')
            best_acc = train_info (model,checkpoint_path,epoch,train_loss,train_acc,valid_loss,valid_acc,elapsed,best_acc,valid_y,pred)
    
    elif len (lst_sensor) == 2 and 's2' in lst_sensor and 'spot' in lst_sensor:
        for epoch in range(n_epochs):
            start = time.time()
            train_S2, train_MS, train_Pan, train_y = shuffle(train_S2, train_MS, train_Pan, train_y, random_state=0)
            for batch in tqdm(range(train_iter),disable=not(tqdm_display)):
                batch_s2 = get_batch (train_S2,batch,batch_size)
                batch_ms = get_batch (train_MS,batch,batch_size)
                batch_pan = get_batch (train_Pan,batch,batch_size)
                batch_y = get_batch (train_y,batch,batch_size)
                train_step(model,None,batch_s2,batch_ms,batch_pan,batch_y,loss_fn,distill_loss_fn,optimizer,train_loss,train_acc,lst_sensor,weight,supervision,is_training=True)
                del batch_s2,batch_ms,batch_pan,batch_y
            pred = []
            for batch in tqdm(range(valid_iter),disable=not(tqdm_display)):
                batch_s2 = get_batch (valid_S2,batch,batch_size)
                batch_ms = get_batch (valid_MS,batch,batch_size)
                batch_pan = get_batch (valid_Pan,batch,batch_size)
                batch_y = get_batch (valid_y,batch,batch_size)
                batch_pred = train_step(model,None,batch_s2,batch_ms,batch_pan,batch_y,loss_fn,distill_loss_fn,optimizer,valid_loss,valid_acc,lst_sensor,weight,supervision,is_training=False)
                del batch_s2,batch_ms,batch_pan,batch_y
                pred.append(batch_pred)
            pred = np.hstack(pred)
            stop = time.time()
            elapsed = stop - start
            if train_hist:
                save_history (history_dic, epoch+1, train_loss.result().numpy(), train_acc.result().numpy(), 'Training')
                save_history (history_dic, epoch+1, valid_loss.result().numpy(), valid_acc.result().numpy(), 'Validation')
            best_acc = train_info (model,checkpoint_path,epoch,train_loss,train_acc,valid_loss,valid_acc,elapsed,best_acc,valid_y,pred)
    
    elif len (lst_sensor) == 1 and 's1' in lst_sensor:
        for epoch in range(n_epochs):
            start = time.time()
            train_S1, train_y = shuffle(train_S1, train_y, random_state=0)
            for batch in tqdm(range(train_iter),disable=not(tqdm_display)):
                batch_s1 = get_batch (train_S1,batch,batch_size)
                batch_y = get_batch (train_y,batch,batch_size)
                train_step(model,batch_s1,None,None,None,batch_y,loss_fn,None,optimizer,train_loss,train_acc,lst_sensor,None,None,is_training=True)
                del batch_s1,batch_y
            pred = []
            for batch in tqdm(range(valid_iter),disable=not(tqdm_display)):
                batch_s1 = get_batch (valid_S1,batch,batch_size)
                batch_y = get_batch (valid_y,batch,batch_size)
                batch_pred = train_step(model,batch_s1,None,None,None,batch_y,loss_fn,None,optimizer,valid_loss,valid_acc,lst_sensor,None,None,is_training=False)
                del batch_s1,batch_y
                pred.append(batch_pred)
            pred = np.hstack(pred)
            stop = time.time()
            elapsed = stop - start
            if train_hist:
                save_history (history_dic, epoch+1, train_loss.result().numpy(), train_acc.result().numpy(), 'Training')
                save_history (history_dic, epoch+1, valid_loss.result().numpy(), valid_acc.result().numpy(), 'Validation')
            best_acc = train_info (model,checkpoint_path,epoch,train_loss,train_acc,valid_loss,valid_acc,elapsed,best_acc,valid_y,pred)

    elif len (lst_sensor) == 1 and 's2' in lst_sensor:
        for epoch in range(n_epochs):
            start = time.time()
            train_S2, train_y = shuffle(train_S2, train_y, random_state=0)
            for batch in tqdm(range(train_iter),disable=not(tqdm_display)):
                batch_s2 = get_batch (train_S2,batch,batch_size)
                batch_y = get_batch (train_y,batch,batch_size)
                train_step(model,None,batch_s2,None,None,batch_y,loss_fn,None,optimizer,train_loss,train_acc,lst_sensor,None,None,is_training=True)
                del batch_s2,batch_y
            pred = []
            for batch in tqdm(range(valid_iter),disable=not(tqdm_display)):
                batch_s2 = get_batch (valid_S2,batch,batch_size)
                batch_y = get_batch (valid_y,batch,batch_size)
                batch_pred = train_step(model,None,batch_s2,None,None,batch_y,loss_fn,None,optimizer,valid_loss,valid_acc,lst_sensor,None,None,is_training=False)
                del batch_s2,batch_y
                pred.append(batch_pred)
            pred = np.hstack(pred)
            stop = time.time()
            elapsed = stop - start
            if train_hist:
                save_history (history_dic, epoch+1, train_loss.result().numpy(), train_acc.result().numpy(), 'Training')
                save_history (history_dic, epoch+1, valid_loss.result().numpy(), valid_acc.result().numpy(), 'Validation')
            best_acc = train_info (model,checkpoint_path,epoch,train_loss,train_acc,valid_loss,valid_acc,elapsed,best_acc,valid_y,pred)

    elif len (lst_sensor) == 1 and 'spot' in lst_sensor:
        for epoch in range(n_epochs):
            start = time.time()
            train_MS, train_Pan, train_y = shuffle(train_MS, train_Pan, train_y, random_state=0)
            for batch in tqdm(range(train_iter),disable=not(tqdm_display)):
                batch_ms = get_batch (train_MS,batch,batch_size)
                batch_pan = get_batch (train_Pan,batch,batch_size)
                batch_y = get_batch (train_y,batch,batch_size)
                train_step(model,None,None,batch_ms,batch_pan,batch_y,loss_fn,None,optimizer,train_loss,train_acc,lst_sensor,None,None,is_training=True)
                del batch_ms,batch_pan,batch_y
            pred = []
            for batch in tqdm(range(valid_iter),disable=not(tqdm_display)):
                batch_ms = get_batch (valid_MS,batch,batch_size)
                batch_pan = get_batch (valid_Pan,batch,batch_size)
                batch_y = get_batch (valid_y,batch,batch_size)
                batch_pred = train_step(model,None,None,batch_ms,batch_pan,batch_y,loss_fn,None,optimizer,valid_loss,valid_acc,lst_sensor,None,None,is_training=False)
                del batch_ms,batch_pan,batch_y
                pred.append(batch_pred)
            pred = np.hstack(pred)
            stop = time.time()
            elapsed = stop - start
            if train_hist:
                save_history (history_dic, epoch+1, train_loss.result().numpy(), train_acc.result().numpy(), 'Training')
                save_history (history_dic, epoch+1, valid_loss.result().numpy(), valid_acc.result().numpy(), 'Validation')
            best_acc = train_info (model,checkpoint_path,epoch,train_loss,train_acc,valid_loss,valid_acc,elapsed,best_acc,valid_y,pred)
    
    if train_hist:
        df = pd.DataFrame.from_dict(history_dic)
        df.to_csv( os.path.join(os.path.dirname(checkpoint_path),os.path.basename(checkpoint_path).replace('model','history')+'.csv') ,index=False)