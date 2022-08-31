# RTDL Models

This is a minimal working version of [1]. To use the Transformer or the Resnet, install the required 
packages and use one of the example files available in the directory "examples" (ft_transformer_{acotsp, lkh}.toml and 
rest_{acotsp, lkh}.toml). To install the required packages, execute the following commands inside the directory name "rtdl"::
```bash
export PROJECT_DIR=$(pwd)

conda install pytorch==1.7.1 torchvision==0.8.2

pip uninstall skranger

pip install -r requirements.txt

conda env config vars set PYTHONPATH=${PYTHONPATH}:${PROJECT_DIR}
conda env config vars set PROJECT_DIR=${PROJECT_DIR}
conda env config vars set LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}

conda deactivate
conda activate oracle
```

You also have to unpack the directory with the model weights:
```bash
wget https://www.inf.ufrgs.br/~gudelazeri/output.tgz
tar zxvf output.tgz
```

Beware of two things:
- These models cannot be directly trained on new datasets, nor be used in new scenarios. Their only purpose is the 
reproducibility of the results of [2]. Their weights are fixed and can be found in the directory "output".
- Due to an incompatibility of versions between the Skranger package required by CSMTOA and the Scikit-learn package required by RTDL, it's necessary to uninstall Skranger before installing the correct version of Sklearn.

---
[1] https://github.com/allbits/rtdl  
[2] TBD

