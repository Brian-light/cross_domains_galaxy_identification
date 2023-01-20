# PHYS449GroupProject

- members: Gabriel Carvalho, Urja Nandivada, Agniya Pobereshnikova, Brian Ruan

## Data Source

Data used in the original paper can be found in:

`https://zenodo.org/record/4507941`

Download the files below from the above website.

- `SimSim_SOURCE_X_Illustris2_pristine.npy`
- `SimSim_SOURCE_y_Illustris2_pristine.npy`
- `SimSim_TARGET_X_Illustris2_noisy.npy`
- `SimSim_TARGET_y_Illustris2_noisy.npy`
- `SimReal_SOURCE_X_Illustris0.npy`
- `SimReal_SOURCE_y_Illustris0.npy`
- `SimReal_TARGET_X_postmergers_SDSS.npy`
- `SimReal_TARGET_y_postmergers_SDSS.npy`

Without changing their names, place them into the data folder.

## File Structure

- `Adversarial`: files for training model using adversarial network
- `MMD`: files for training model using maximum mean discrepancy
- `data`: this is where you should put all your training data
- `data_sample`: a small set of training data, with x the images and y the labels
- `image_examples`: example images extracted from the dataset
- `mosaic`: script used to obtain image examples from the dataset 
- `no_DA`: files for training model using no domain adaptation method
- `normalization`: script used to normalize images from the original dataset
- `tSNE`: script for plotting tSNE plot
- `tSNE_plots`: contains tSNE plots produced by MMD and no DA

## How to

### Google Colab
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="45">

This project is now fully supported on Google Colab. To run this project with Colab:

1.Download the repository
2.Go to Google Colab -> File -> Open notebook
3.Select the file `colab_notebook.ipynb` from the repository

results can be found in the 'results' folder on the top level.

### Native machines

This project is only tested on Windows, support for other platforms is not guaranteed

1.First, download the files from the data source listed above.

2.Place the files in the data folder on the top level.

3.Install the python dependencies below:
- `numpy`
- `torch`
- `torchvision`
- `matplotlib`
- `scikit-learn`

4.Use the script in `noramlization` to normalize both the source domain and target domain images

5.The three different training methods (no_DA, adversarial and MMD) can be found in their respected folders

6.Adjust hyperparameters inside `param.json` file

7.Execute `main.py` with default setting by calling
```sh
python main.py 
```
For custome run, use 
```sh
usage: main.py [param] [res_path]
```

with positional arguments:
  
  -param: path to hyperparameter json
  
  -res_path: results directory
  
6.Some module may vary, be sure to checkout the readme in each folder.
