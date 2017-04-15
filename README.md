# VOCSeg 

VOCSeg performs segmentation of voc2012 by utilizing an FCN based model. 

VOCSeg model is firstly cloned from [KittiSeg model](https://github.com/MarvinTeichmann/KittiSeg.git) 

Based on KittiSeg model, we make some modifications for multi-class segmentation.

KittiSeg model which is achieved [first place](http://www.cvlibs.net/datasets/kitti/eval_road_detail.php?result=ca96b8137feb7a636f3d774c408b1243d8a6e0df) on the Kitti Road Detection Benchmark at submission time. Check out their [paper](https://arxiv.org/abs/1612.07695) for a detailed model description.

The repository contains code for training, evaluating and visualizing semantic segmentation in TensorFlow. It is build to be compatible with the [TensorVision](http://tensorvision.readthedocs.io/en/master/user/tutorial.html#workflow) back end which allows to organize experiments in a very clean way. 


## Requirements

The code requires [Tensorflow 1.0](https://www.tensorflow.org/install/) as well as the following python libraries: 

* matplotlib
* numpy
* Pillow
* scipy
* commentjson

Those modules can be installed using: `pip install numpy scipy pillow matplotlib commentjson` or `pip install -r requirements.txt`.


## Setup

1. Clone this repository: `git clone https://github.com/lxh-123/VOCSeg.git`
2. Initialize all submodules: `git submodule update --init --recursive`
3. [Optional] Download VOC2012 Data:
    1. Retrieve voc data url here: [http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)
    2. Call `python download_data.py --voc_url URL_YOU_RETRIEVED`
    
Step 3 is only required if you want to train your own model using `train.py` or bench a model agains the official evaluation score `evaluate.py`. Also note, that I recommend using `download_data.py` instead of downloading the data yourself. The script will also extract and prepare the data. See Section [Manage data storage](README.md#manage-data-storage) if you like to control where the data is stored.

##### To update an existing installation do:

1. Pull all patches: `git pull`
2. Update all submodules: `git submodule update --init --recursive`

If you forget the second step you might end up with an inconstant repository state. 

## Tutorial

### Getting started

Run: `python evaluate.py` to evaluate a trained model. 

Run: `python train.py --hypes hypes/VOCSeg.json` to train a model using voc2012 Data.


### Manage Data Storage

VOCSeg allows to separate data storage from code. This is very useful in many server environments. By default, the data is stored in the folder `VOCSeg/DATA` and the output of runs in `VOCSeg/RUNS`. This behaviour can be changed by setting the bash environment variables: `$TV_DIR_DATA` and `$TV_DIR_RUNS`.

Include  `export TV_DIR_DATA="/MY/LARGE/HDD/DATA"` in your `.profile` and the all data will be downloaded to `/MY/LARGE/HDD/DATA/data_road`. Include `export TV_DIR_RUNS="/MY/LARGE/HDD/RUNS"` in your `.profile` and all runs will be saved to `/MY/LARGE/HDD/RUNS/VOCSeg`

### RUNDIR and Experiment Organization

VOCSeg helps you to organize large number of experiments. To do so the output of each run is stored in its own rundir. Each rundir contains:

* `output.log` a copy of the training output which was printed to your screen
* `tensorflow events` tensorboard can be run in rundir
* `tensorflow checkpoints` the trained model can be loaded from rundir
* `[dir] images` a folder containing example output images. `image_iter` controls how often the whole validation set is dumped
* `[dir] model_files` A copy of all source code need to build the model. This can be very useful of you have many versions of the model.

To keep track of all the experiments, you can give each rundir a unique name with the `--name` flag. The `--project` flag will store the run in a separate subfolder allowing to run different series of experiments. As an example, `python train.py --project batch_size_bench --name size_5` will use the following dir as rundir:  `$TV_DIR_RUNS/VOCSeg/batch_size_bench/size_5_VOCSeg_2017_02_08_13.12`.

The flag `--nosave` is very useful to not spam your rundir.

### Modifying Model & Train on your own data

The model is controlled by the file `hypes/VOCSeg.json`. Modifying this file should be enough to train the model on your own data and adjust the architecture according to your needs. A description of the expected input format can be found [here](docu/inputs.md).


For advanced modifications, the code is controlled by 5 different modules, which are specified in `hypes/VOCSeg.json`.

```
"model": {
   "input_file": "../inputs/voc_seg_input.py",
   "architecture_file" : "../encoder/fcn8_vgg.py",
   "objective_file" : "../decoder/voc_multiloss.py",
   "optimizer_file" : "../optimizer/generic_optimizer.py",
   "evaluator_file" : "../evals/voc_eval.py"
},
```

Those modules operate independently. This allows easy experiments with different datasets (`input_file`), encoder networks (`architecture_file`), etc. Also see [TensorVision](http://tensorvision.readthedocs.io/en/master/user/tutorial.html#workflow) for a specification of each of those files.




## Utilize TensorVision backend

VOCSeg is build on top of the TensorVision [TensorVision](https://github.com/TensorVision/TensorVision) backend. TensorVision modularizes computer vision training and helps organizing experiments. 


To utilize the entire TensorVision functionality install it using 

`$ cd VOCSeg/submodules/TensorVision` <br>
`$ python setup install`

Now you can use the TensorVision command line tools, which includes:

`tv-train --hypes hypes/VOCSeg.json` trains a json model. <br>
`tv-continue --logdir PATH/TO/RUNDIR` trains the model in RUNDIR, starting from the last saved checkpoint. Can be used for fine tuning by increasing `max_steps` in `model_files/hypes.json` .<br>
`tv-analyze --logdir PATH/TO/RUNDIR` evaluates the model in RUNDIR <br>


## Useful Flags & Variabels

Here are some Flags which will be useful when working with VOCSeg and TensorVision. All flags are available across all scripts. 

`--hypes` : specify which hype-file to use <br>
`--logdir` : specify which logdir to use <br>
`--gpus` : specify on which GPUs to run the code <br>
`--name` : assign a name to the run <br>
`--project` : assign a project to the run <br>
`--nosave` : debug run, logdir will be set to `debug` <br>

In addition the following TensorVision environment Variables will be useful:

`$TV_DIR_DATA`: specify meta directory for data <br>
`$TV_DIR_RUNS`: specify meta directory for output <br>
`$TV_USE_GPUS`: specify default GPU behaviour. <br>

On a cluster it is useful to set `$TV_USE_GPUS=force`. This will make the flag `--gpus` mandatory and ensure, that run will be executed on the right GPU.

## Questions?

Please have a look into the [FAQ](docu/FAQ.md). Also feel free to open an issue to discuss any questions not covered so far. 

# Citation

If you benefit from this code, please cite the original paper:

```
@article{teichmann2016multinet,
  title={MultiNet: Real-time Joint Semantic Reasoning for Autonomous Driving},
  author={Teichmann, Marvin and Weber, Michael and Zoellner, Marius and Cipolla, Roberto and Urtasun, Raquel},
  journal={arXiv preprint arXiv:1612.07695},
  year={2016}
}
```

