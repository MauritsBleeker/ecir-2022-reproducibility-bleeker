This is the PyTorch code repository for the ECIR 2022 reproducibility track submission: "Do Lessons from Metric Learning Generalize to Image-Caption Retrieval?" By Maurits Bleeker and Maarten de Rijke.

This project is built on top of/is using  the code of [VSE++](https://github.com/fartashf/vsepp), [VSRN](https://github.com/KunpengLi1994/VSRN/), and some work of [SmoothAP](https://github.com/Andrew-Brown1/Smooth_AP). 

### Install environment

Make sure to install anaconda.

```bash
$ conda env create -f environment.yml
```

### Folder structure 

In the root of this repository, create the following folders.

```bash
$ mkdir data
$ mkdir out
```

### Data formatting and loading 

For the [VSRN](https://github.com/KunpengLi1994/VSRN/) method, we used the all data (train/test/val + vocab) as [provided](https://github.com/KunpengLi1994/VSRN/#download-datahttps://github.com/KunpengLi1994/VSRN/#download-data) by the authors. 

For VSE++ we have reformatted the data to a pickle file and changed the data loader. We did this to improve the training time. 
The data for the Flickr30k dataset (train/test/val + vocab)  can be downloaded [here](https://surfdrive.surf.nl/files/index.php/s/qnWabw1G5IqwARm?path=%2Fdata%2Fvsepp%2FVSEPP%2Ff30k). 
The MS-COCO dataset is too big to upload in one file. However, if the [MS-Coco](https://cocodataset.org/#home) caption dataset is formatted by using the format below, using [dataset_coco.json](https://surfdrive.surf.nl/files/index.php/s/qnWabw1G5IqwARm?path=%2Fdata), then the experiments can be executed.
In `/notebooks` we provide a example script on how to format the MS-COCO data. Be aware that this is not the script we have used for our experiments, so there might be some differences.
The vocab file for MS-COCO can be found [here](https://surfdrive.surf.nl/files/index.php/s/qnWabw1G5IqwARm?path=%2Fdata%2Fvocab_vsrn)



```
{
'images': {img_id: {
    'image': str(binary_img_file),
    'filename': str(file_namem), 
    'caption_ids': list(id1, ...)
    }
},
'captions': {
    'caption': caption['raw']
    'imgid': caption['imgid']
    'tokens': caption['tokens']
    'filename'
  },
}
```
### Run experiments/train model

This research has been conducted by using a compute cluster in combination with SLURM. In the folder `/jobs`, we give all job files used to run the experiments. To run the job files yourself, change all the `{path to repo}` and `{path to vocab}` into the correct folder or use an environment variable. In the hyper-parameter file in `/jobs/{dataset}/papper_experiments/baseline/*hyper_params.txt`, all the hyper-parameters are given.

To start training (one experiment) from the command line, use the following exaples:

```bash
$ python vsepp/train.py --data_name=coco --data_path={path to repo}/data/vsepp/coco -vocab_path={path to repo}/vsepp/coco --cnn_type=resnet50 --workers=10 --experiment_name=coco_triplet_max --criterion triplet --max_violation --logger_name {path to repo}/out/vsepp/out/coco/paper_experiments/triplet_max/0
```

```bash
$ python VSRN/train.py --data_name=coco_precomp --data_path={path to repo}/data/vsrn/coco -vocab_path={path to repo}/vsrn/coco --cnn_type=resnet50 --workers=10 --max_len 60 -experiment_name=coco_triplet_max --criterion triplet --max_violation --logger_name {path to vocab}/out/vsrn/out/coco/paper_experiments/triplet_max/0 --lr_update 15
```

We run each experiment five times. The model files for each experiment are stored at `{path to repo}/out/{method}/out/{dataset}/paper_experiments/{loss function}/{0,1,3,4}`

### Training parameters

In in the `*.job` and `*hyper_params.txt` you can set the following parameters. 
- `--data_name`: Name of the datasets used in the work. For VSE++ use `f30k` and `coco`, for VSRN use `f30k_precomp` and `coco_precomp`.
- `--data_path`: Path where datasets are stored.
- `--vocab_path`: Path where vocab files are stored.
- `--max_violation`: Only use the maximum violating triplet in the batch per query. I.e. Triplet loss SH. Only possible in combination with triplet loss.
- `--cnn_type`: For VSE++ only, use `resnet50`.
- `--workers`: Number of workers for the DataLoader, default is 10.
- `--max_len`: For VSRN only, max-length of the caption.
- `--experiment_name`: Name of the experiment, for `wandb`. Options used: `triplet_max`, `triplet_max`, `triplet`, `ntxent`, `smoothap`.
- `--criterion`: Loss functions used for training. Options used: `triplet`, `ntxent`, `ranking` (SmoothAP).
- `--ranking_based`: Flag to indicate that we optimize a ranking. I.e., we want to use all the captions that match to an image.
- `--tau`: Tau hyper-parameter value for SmoothAP and NTXent
- `--logger_name`: Path where all the log-files are stored. Use the following format `{path to repo}/out/{method}/out/{dataset}/paper_experiments/{loss }/{experiment number}`. For {loss} use `triplet_max`, `triplet`, `ntxent`, `ranking`.
- `--num_epochs`: Number of training epochs.
- `--lr_update`: Drop lr after n epochs with factor 0.1.

The rest of the method specific parameters can be found in `vseppp/train.py` and `VSRN/train.py`.

### Run evaluation

In `/evaluation`, the notebooks are provided to generate all the results tables from Sections 3 and 5. 
The file `Evaluation notebook.ipynb` is used to generate Table 3.

The COCOs results are generated by using `Gradient counting-Coco-VSE++.ipynb`, `Gradient counting-Coco-VSRN.ipynb`, `Gradient counting-F30k-VSE++-.ipynb` and `Gradient counting-F30k-VSRN.ipynb`. 

In total, we have trained 5 * 4 * 2 * 2 models. It is not feasible to share all the parameter files. [Here](https://surfdrive.surf.nl/files/index.php/s/EWjyBatYA60L0xx) we give all the model checkpoints for experiment 1.2, 1.6, 2.2, 2.6 (the experiments with Triplet loss SH/Triplet loss max).

Be aware that the checkpoint files contain a copy of all the hyper-parameters that have been used for training and that some of the paths in this config file might point to folders that have been set while training the models. This should be changed.

### Citation

If you use this code to produce results for your scientific work please refer to our ECIR 2022 paper:
```
@inproceedings{bleeker2022lessons,
  title={Do Lessons from Metric Learning Generalize to Image-Caption Retrieval?},
  author={Bleeker, Maurits and Rijke, Maarten de},
  booktitle={European Conference on Information Retrieval},
  pages={535--551},
  year={2022},
  organization={Springer}
}
```
