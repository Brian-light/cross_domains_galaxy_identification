# Run
Make sure you normalize the images first

Adjust the hyperparameters in the param file

Then run with the same param file

Execute `main.py` with default setting by calling
```sh
python main.py 
```
For custome run, use 
```sh
usage: main.py [param] [res_path]
```

with positional arguments:
  
  -param: path to hyperparameter json(default: param.json)
  
  -res_path: results directory(default: results)
  

# Parameters

```sh
"general":{
		"seed": 1
"data":{
		"path": "C:/phys project/PHYS449GroupProject-main/PHYS449GroupProject-main/data/",
		"sim_sim_src_x_file": "SimSim_SOURCE_X_Illustris2_pristine.npy",
		"sim_sim_src_y_file": "SimSim_SOURCE_y_Illustris2_pristine.npy",
		"sim_sim_tar_x_file": "SimSim_TARGET_X_Illustris2_noisy.npy",
		"sim_sim_tar_y_file": "SimSim_TARGET_y_Illustris2_noisy.npy",
		"sim_real_src_x_file": "SimReal_SOURCE_X_Illustris0.npy",
		"sim_real_src_y_file": "SimReal_SOURCE_y_Illustris0.npy",
		"sim_real_tar_x_file": "SimReal_TARGET_x_postmergers_SDSS.npy",
		"sim_real_tar_y_file": "SimReal_TARGET_y_postmergers_SDSS.npy",
		"n_test_data": 3000,
		"num_classes": 2,
		"test_batch_size":3000,
		"use_real_dataset": true,
		"train_on_target": false,
		"do_transfer_model": false,
		"transfer_model": "G:/repo/PHYS449GroupProject/Adversarial/results_6/DeepMerge_net.pt"
	},
"exec":{
		"num_epochs": 200,
		"mode": 1,
		"use_scheduler": false		
	}
    
```
`seed`: seed for data shuffling

`path`: the directory where you stored you data

`sim_sim_src_x_file`: file name for source domain image for sim-sim

`sim_sim_src_y_file`: file name for source domain label for sim-sim

`sim_sim_tar_x_file`: file name for target domain image for sim-sim

`sim_sim_tar_y_file`: file name for target domain label for sim-sim

"sim_real_src_x_file`: file name for source domain image for sim-real

`sim_real_src_y_file`: file name for source domain label for sim-real

`sim_real_tar_x_file`: file name for target domain image for sim-real

`sim_real_tar_y_file`: file name for target domain label for sim-real

`n_test_data`: number of samples use for testing

`num_classes`: number of class to classify, always use 2

`batch_size": batch number for training

`test_batch_size`: batch number for test set

`use_real_dataset`: if false, use sim-sim dataset for training. if true, use sim-real dataset for training

`do_transfer_model`: if false, performing training as usual. if true, skip training and load trained model

`transfer model`: path for trained model to be loaded

`num_epochs`: number of epochs to train

`display_epochs`: always use 1

`use_scheduler`: if true, enable scheduler during training. if false, disable it


