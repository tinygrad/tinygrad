## Steps to download and verify data

1. Download the data
   
    To download the data please follow the instructions:
    ```bash
    mkdir raw-data-dir
    cd raw-data-dir
    git clone https://github.com/neheller/kits19
    cd kits19
    pip3 install -r requirements.txt
    python3 -m starter_code.get_imaging
    ```
    This will download the original, non-interpolated data to `raw-data-dir/kits19/data`

2. Preprocess the dataset.
    
    The data preprocessing script is called `preprocess_dataset.py`. All the required hyperparameters are already set. All you need to do is to invoke the script with correct paths:
    ```bash
    python3 preprocess_dataset.py --data_dir /raw_data --results_dir /data
    ```
   
    The script will preprocess each volume and save it as a numpy array at `/data`. It will also display some statistics like the volume shape, mean and stddev of the voxel intensity. Also, it will run a checksum on each file comparing it with the source.
