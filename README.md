# S1S2VHSR
Multi-sensor and Multi-scale data fusion for land cover mapping through Convolutional Neural Networks (CNN). 

This repository supports a paper we have submitted. The study assesses the fusion of Sentinel-1 (S1) and Sentinel-2 (S2) satellite image time series in addition to a Very High Spatial Resolution (VHSR) SPOT image for land cover mapping via a 3 branch CNN architecture. The study was carried out on Reunion island and Dordogne department study sites.

## Prerequisites

The code relies on Pyhton 3.7.6. The CNN models were implemented with Tensorflow 2. 

## Examples 

### Running the main architecture model

- fusion of S1, S2 and SPOT

```
python main.py s1_path s2_path ms_path pan_path gt_path
```

- See help for descriptions `python main.py -h/--help`

### Running ablation models 

- available cases: S1 and S2, S2 and SPOT, S1, S2 and SPOT. S1 and S2 data can be leveraged with 1D, 2D or 3D CNN. SPOT data are leveraged only with a 2D CNN.

- example for S1 with 2D-CNN and S2 with 1D-CNN `python main.py s1_path s2_path ms_path pan_path gt_path -s s1-2D s2-1D`

- example for S2 with 1D-CNN and SPOT `python main.py s1_path s2_path ms_path pan_path gt_path -s s2-1D spot`

### Change some hyperparameters

- See help for all available options

- batch size: `-bs` default value `256`

- learning rate: `-lr` default value `1e-4`

- number of epochs: `-ep` default value `1000`

- control the number of per source features: `-nf` default value `128` for 256 (128 x 2) features per source

- how to fuse the per source feature: `-f` default value `concat` for Concatenation  (`add` Addition also available)
