# nfl-field-goal-model

## Setting up the environment

### 1. Create the conda environment
First, create the conda environment from the `environment.yml` file:
```bash
conda env create -f environment.yml
```

### 2. Activate the environment
Activate the newly created environment:
```bash
conda activate nfl-fg
```

### 3. Create a Jupyter kernel
Create a Jupyter kernel for this environment so you can use it in Jupyter notebooks:
```bash
python -m ipykernel install --user --name nfl-fg
```

### 4. Verify the installation
You can verify that the kernel was created successfully by listing all available kernels:
```bash
jupyter kernelspec list
```

### 5. Start Jupyter
Open one of the notebooks and select the `nfl-fg` kernel.
