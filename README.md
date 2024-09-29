# ReparamCAD: Zero-shot CAD Program Re-Parameterization for Interactive Manipulation

Public code release for "ReparamCAD: Zero-shot CAD Program Re-Parameterization for Interactive Manipulation, SIGGRAPH Asia 2023 Conference Papers, authored by Milin Kodnongbua*, Benjamin Jones*, Maaz Bin Safeer Ahmad, Vladimir Kim, and Adriana Schulz.

You are permitted to use/edit/recompose the code for non-commercial purposes only.

## Installation

1. Create a conda environment with:
```sh
conda env create -f environment.yml
conda activate rpcad
```

2. Get [CUDA Toolkit 11](https://developer.nvidia.com/cuda-11-8-0-download-archive).

3. Download `sd-v1-4.ckpt` from [https://huggingface.co/CompVis/stable-diffusion-v-1-4-original](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original) to `weights/sd-v1-4.ckpt`, or:

```sh
cd weights
wget https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt
```

## Usage

### Step 1: Running the generation process for each prompt

```sh
python run.py name="example" prompt="a chair" init.model="chair_arm" task="sd"
```

- Stable diffusion requires a GPU with at least 12GB of VRAM.
- The output will be saved to the `output` folder.
- Supported `init.model`
    - `chair_arm`, `table`, `car`, `camera`, `bottle`
    - You can append `_rot` to enable optimizing for rotation

### Step 2: Running the constraint discovery

- We use a mesh boolean engine from [Cherchi et. al.](https://github.com/gcherchi/InteractiveAndRobustMeshBooleans). Please clone their repository to a sibling folder to ReparamCAD and follow their instruction to compile it. (See `rpcad/mesh_boolean.py` for the wrapper to the engine).


```
python TODO
```

## Development

1. Install `pre-commit` to format Python code.

```
pip install pre-commit
pre-commit install
```

2. Add a filter to remove notebook output

```
git config --local filter.remove-notebook-ouput.clean "jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --to=notebook --stdin --stdout --log-level=ERROR"
```

3. To run the scripts in Hyak, load my fork of [`simple_slurm`](https://github.com/milmillin/simple_slurm).

```
pip install -e git+https://github.com/milmillin/simple_slurm.git#egg=simple_slurm
```

## Citation

TODO

