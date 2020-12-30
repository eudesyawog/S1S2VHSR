import sys
import os
import argparse
from pathlib import Path
import numpy as np
from src.model import Model_S1S2SPOT, Model_S1S2, Model_S2SPOT, Model_S1, Model_S2, Model_SPOT
from src.utils import format_cnn1d,format_cnn2d,format_y
from src.train import run
from src.test import restore
from src.embedding import getEmbedding

if __name__ == '__main__':
    
    # Parsing arguments
    if len(sys.argv) == 1:
        print ('Usage: python '+sys.argv[0]+' s1_path s2_path ms_path pan_path gt_path [options]' )
        print ('Help: python '+sys.argv[0]+' -h/--help')
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('s1_path',help='Path to Sentinel-1 SITS',type=str)
    parser.add_argument('s2_path',help='Path to Sentinel-2 SITS',type=str)
    parser.add_argument('ms_path',help='Path to SPOT Multispectral',type=str)
    parser.add_argument('pan_path',help='Path to SPOT Panchromatic',type=str)
    parser.add_argument('gt_path',help='Path to ground truth',type=str)
    parser.add_argument('num_split',help='Number of split to use',type=str)
    parser.add_argument('-out','--out_path',help='Output path for model and results',type=str)
    parser.add_argument('-s','--sensor',dest='sensor',help='Input sensor data and dimension to consider for Sentinel',nargs='+',choices=['s1-1D','s1-2D','s1-3D','s2-1D','s2-2D','s2-3D','spot'],default=['s1-2D','s2-1D','spot'])
    parser.add_argument('-bs','--batch_size',dest='batch_size',help='Batch size',default=256,type=int)
    parser.add_argument('-ep','--num_epochs',dest='num_epochs',help='Number of training epochs',default=1000,type=int)
    parser.add_argument('-lr','--learning_rate',dest='learning_rate',help='Learning rate',default=1E-4,type=float)
    parser.add_argument('-drop','--dropout_rate',dest='dropout_rate',help='Dropout rate',default=0.5,type=float)
    parser.add_argument('-w','--weight',dest='weight',help='Weight for auxiliary classifiers',default=0.3,type=float)
    parser.add_argument('-f','--fusion',dest='fusion',help='How to fuse per source features ?',choices=['add','concat'],default='concat')
    parser.add_argument('-nf','--num_feat',dest='num_feat',help='Number of per source features',default=128,type=float)
    parser.add_argument('-tqdm',dest='tqdm',help='Display tqdm progress bar',default=False,type=bool)
    args = parser.parse_args()

    # Get argument values
    s1_path = args.s1_path
    s2_path = args.s2_path
    ms_path = args.ms_path
    pan_path = args.pan_path
    gt_path = args.gt_path
    sensor = args.sensor
    lst_sensor = [el.split('-')[0] for el in sensor]
    if len(sensor) > 3 or len(lst_sensor)!=len(set(el.split('-')[0] for el in sensor)):
        sys.exit(f"{os.path.basename(__file__)}: error: Too much input sensor data or repeated input source")

    if not args.out_path is None :
        out_path = args.out_path
    else:
        out_path = os.path.join('models',f'model_{"_".join(el.upper() for el in sensor)}')
    batch_size = args.batch_size
    n_epochs = args.num_epochs
    lr = args.learning_rate
    drop = args.dropout_rate
    weight = args.weight
    fusion = args.fusion
    num_feat = args.num_feat
    n_split = args.num_split
    tqdm_disable = args.tqdm
    
    # Create output path if does not exist
    Path(out_path).mkdir(parents=True, exist_ok=True) 

    # Load Training and Validation set
    train_y = format_y(gt_path+f'/Training/Ground_truth_Training_split_{n_split}.npy')#'/train_gt.npy')
    print ('Training GT:',train_y.shape)
    valid_y = format_y(gt_path+f'/Validation/Ground_truth_Validation_split_{n_split}.npy')#'/valid_gt.npy')
    print ('Validation GT:', valid_y.shape)
    n_classes = len(np.unique(train_y))
    print ('Number of classes:',n_classes)
    
    if 's1-1D' in sensor :
        train_S1 = format_cnn1d(s1_path+f'/Training/Sentinel-1_Training_split_{n_split}.npy')#'/train_S1.npy')
        print ('Training S1:',train_S1.shape)
        valid_S1 = format_cnn1d(s1_path+f'/Validation/Sentinel-1_Validation_split_{n_split}.npy')#'/valid_S1.npy')
        print ('Validation S1:',valid_S1.shape)
    elif 's1-2D' in sensor :
        train_S1 = format_cnn2d(s1_path+f'/Training/Sentinel-1_Training_split_{n_split}.npy')#'/train_S1.npy')
        print ('Training S1:',train_S1.shape)
        valid_S1 = format_cnn2d(s1_path+f'/Validation/Sentinel-1_Validation_split_{n_split}.npy')#'/valid_S1.npy')
        print ('Validation S1:',valid_S1.shape)
    elif 's1-3D' in sensor :
        train_S1 = np.load(s1_path+f'/Training/Sentinel-1_Training_split_{n_split}.npy')#'/train_S1.npy')
        print ('Training S1:',train_S1.shape)
        valid_S1 = np.load(s1_path+f'/Validation/Sentinel-1_Validation_split_{n_split}.npy')#'/valid_S1.npy')
        print ('Validation S1:',valid_S1.shape)
    else:
        train_S1,valid_S1 = (None,None)
    
    if 's2-1D' in sensor :
        train_S2 = format_cnn1d(s2_path+f'/Training/Sentinel-2_Training_split_{n_split}.npy')#'/train_S2.npy')
        print ('Training S2:',train_S2.shape)
        valid_S2 = format_cnn1d(s2_path+f'/Validation/Sentinel-2_Validation_split_{n_split}.npy')#'/valid_S2.npy')
        print ('Validation S2:',valid_S2.shape)
    elif 's2-2D' in sensor :
        train_S2 = format_cnn2d(s2_path+f'/Training/Sentinel-2_Training_split_{n_split}.npy')#'/train_S2.npy')
        print ('Training S2:',train_S2.shape)
        valid_S2 = format_cnn2d(s2_path+f'/Validation/Sentinel-2_Validation_split_{n_split}.npy')#'/valid_S2.npy')
        print ('Validation S2:',valid_S2.shape)
    elif 's2-3D' in sensor :
        train_S2 = np.load(s2_path+f'/Training/Sentinel-2_Training_split_{n_split}.npy')#'/train_S2.npy')
        print ('Training S2:',train_S2.shape)
        valid_S2 = np.load(s2_path+f'/Validation/Sentinel-2_Validation_split_{n_split}.npy')#'/valid_S2.npy')
        print ('Validation S2:',valid_S2.shape)
    else:
        train_S2,valid_S2 = (None,None)

    if 'spot' in sensor :
        train_MS = format_cnn2d(ms_path+f'/Training/Spot-MS_Training_split_{n_split}.npy')#'/train_MS.npy')
        print ('Training MS:',train_MS.shape)
        valid_MS = format_cnn2d(ms_path+f'/Validation/Spot-MS_Validation_split_{n_split}.npy')#'/valid_MS.npy')
        print ('Validation MS:',valid_MS.shape)
        train_Pan = format_cnn2d(pan_path+f'/Training/Spot-P_Training_split_{n_split}.npy')#'/train_Pan.npy')
        print ('Training Pan:',train_Pan.shape)
        valid_Pan = format_cnn2d(pan_path+f'/Validation/Spot-P_Validation_split_{n_split}.npy')#'/valid_Pan.npy')
        print ('Validation Pan:',valid_Pan.shape)
    else:
        train_MS,valid_MS,train_Pan,valid_Pan = (None,None,None,None)

    # Create the Tensorflow model
    if len (sensor) == 3:
        model = Model_S1S2SPOT (drop,n_classes,num_feat,fusion,sensor)
    elif len (sensor) == 2 and 's1' in lst_sensor and 's2' in lst_sensor :
        model = Model_S1S2 (drop,n_classes,num_feat,fusion,sensor)
    elif len (sensor) == 2 and 's2' in lst_sensor and 'spot' in lst_sensor :
        model = Model_S2SPOT (drop,n_classes,num_feat,fusion,sensor)
    elif len (sensor) == 1 and 's1' in lst_sensor :
        model = Model_S1 (drop,n_classes,num_feat,sensor)
    elif len (sensor) == 1 and 's2' in lst_sensor :
        model = Model_S2 (drop,n_classes,num_feat,sensor)
    elif len (sensor) == 1 and 'spot' in lst_sensor :
        model = Model_SPOT (drop,n_classes,num_feat)

    # Learning stage
    checkpoint_path = os.path.join(out_path,f'model_{n_split}') 

    run (model,train_S1,train_S2,train_MS,train_Pan,train_y,
            valid_S1,valid_S2,valid_MS,valid_Pan,valid_y,
                checkpoint_path,batch_size,lr,n_epochs,lst_sensor,weight,tqdm_disable)

    # Load Test set 
    test_y = format_y(gt_path+f'/Test/Ground_truth_Test_split_{n_split}.npy',encode=False)#'/test_gt.npy',encode=False)
    print ('Test GT:',test_y.shape)
    
    if 's1-1D' in sensor :
        test_S1 = format_cnn1d(s1_path+f'/Test/Sentinel-1_Test_split_{n_split}.npy')#'/test_S1.npy')
        print ('Test S1:',test_S1.shape)
    elif 's1-2D' in sensor :
        test_S1 = format_cnn2d(s1_path+f'/Test/Sentinel-1_Test_split_{n_split}.npy')#'/test_S1.npy')
        print ('Test S1:',test_S1.shape)
    elif 's1-3D' in sensor :
        test_S1 = np.load(s1_path+f'/Test/Sentinel-1_Test_split_{n_split}.npy')#'/test_S1.npy')
        print ('Test S1:',test_S1.shape)
    else:
        test_S1 = None

    if 's2-1D' in sensor :
        test_S2 = format_cnn1d(s2_path+f'/Test/Sentinel-2_Test_split_{n_split}.npy')#'/test_S2.npy')
        print ('Test S2:',test_S2.shape)
    elif 's2-2D' in sensor :
        test_S2 = format_cnn2d(s2_path+f'/Test/Sentinel-2_Test_split_{n_split}.npy')#'/test_S2.npy')
        print ('Test S2:',test_S2.shape)
    elif 's2-3D' in sensor :
        test_S2 = np.load(s2_path+f'/Test/Sentinel-2_Test_split_{n_split}.npy')#'/test_S2.npy')
        print ('Test S2:',test_S2.shape)
    else:
        test_S2 = None

    if 'spot' in sensor :
        test_MS = format_cnn2d(ms_path+f'/Test/Spot-MS_Test_split_{n_split}.npy')#'/test_MS.npy')
        print ('Test MS:',test_MS.shape)
        test_Pan = format_cnn2d(pan_path+f'/Test/Spot-P_Test_split_{n_split}.npy')#'/test_Pan.npy')
        print ('Test Pan:',test_Pan.shape)
    else:
        test_MS,test_Pan = (None,None)

    # Inference stage
    result_path = os.path.join(out_path,f'pred_{n_split}.npy')
    restore (model,test_S1,test_S2,test_MS,test_Pan,test_y,batch_size,checkpoint_path,result_path,lst_sensor,tqdm_disable)

    # Get Embedding on test set
    embedding_path = os.path.join(out_path,f'embedding_{n_split}.npy')
    getEmbedding (model,test_S1,test_S2,test_MS,test_Pan,test_y,batch_size,checkpoint_path,embedding_path,lst_sensor,tqdm_disable)