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
  
  -param: path to hyperparameter json default='param.json'
  
  -res_path: results directory default='results'
  

# Parameters

```sh
"data":{
		"data directory": "C:/phys project/PHYS449GroupProject-main/PHYS449GroupProject-main/data/",
		"seed data split": 1,
		"training ratio": 0.7,
		"valid ratio": 0.1,
		"testing ratio": 0.2,
		"sim_sim_src_x_file": "SimSim_SOURCE_X_Illustris2_pristine.npy",
		"sim_sim_src_y_file": "SimSim_SOURCE_y_Illustris2_pristine.npy",
		"sim_sim_tar_x_file": "SimSim_TARGET_X_Illustris2_noisy.npy",
		"sim_sim_tar_y_file": "SimSim_TARGET_y_Illustris2_noisy.npy",
		"sim_real_src_x_file": "SimReal_SOURCE_X_Illustris0.npy",
		"sim_real_src_y_file": "SimReal_SOURCE_y_Illustris0.npy",
		"sim_real_tar_x_file": "SimReal_TARGET_x_postmergers_SDSS.npy",
		"sim_real_tar_y_file": "SimReal_TARGET_y_postmergers_SDSS.npy",
		"use_real_dataset": false,
		"batch_size": 128,
		"do_transfer_model": false,
		"transfer_model": "C:/phys project/PHYS449GroupProject-main/PHYS449GroupProject-main/Adversarial/results_6/DeepMerge_net.pt"
	},
"exec":{
		"num_epochs": 200,
		"mode": 1,
		"use_scheduler": false		
	}
    
```
`data directory`: the directory where you stored you data

`seed data split`: seed used when splitting data into training, validation, and test set

'training ratio`: ratio of the training set

'valid ratio`: ratio of the validation set

`testing ratio`: ratio of the testing set

`sim_sim_src_x_file`: file name for source domain image for sim-sim

`sim_sim_src_y_file`: file name for source domain label for sim-sim

`sim_sim_tar_x_file`: file name for target domain image for sim-sim

`sim_sim_tar_y_file`: file name for target domain label for sim-sim

"sim_real_src_x_file`: file name for source domain image for sim-real

`sim_real_src_y_file`: file name for source domain label for sim-real

`sim_real_tar_x_file`: file name for target domain image for sim-real

`sim_real_tar_y_file`: file name for target domain label for sim-real

`use_real_dataset`: if false, use sim-sim dataset for training. if true, use sim-real dataset for training

`batch_size`: batch number for training

`do_transfer_model`: if false, performing training as usual. if true, do transfer learning

`transfer model`: path for trained model to be loaded

`num_epochs`: number of epochs to train

`mode`: if 1, train with MMD + fisher + EM. Anything else, train with MMD only

`use_scheduler`: if true, enable scheduler during training. if false, disable it


