## Hardware requirements
  - GPU with 11Gb memory is necessary
  - 64Gb RAM memory at least  # actually 32 should be enough but, unfortunately, there are memory leak exists, I still not find one
  - 120Gb free space is required (ssd type partitions is recommended)

***

## Start with docker
  - `git clone git@github.com:i7p9h9/myna-lab-task.git`
  - cd speech2num/
  - prepare docker environment:
  ```bash
  ./local/prepare_for_docker.sh
  ```

***

## To run training process 
  - manage `path.sh`
    - DATASETS_DIR - path to folder with trainig data, next structure expected:
    ```bash
    .
    ├── test-example
    ├── test-example.csv
    ├── train
    ├── train.csv
    └── train-orig
    ```
    - PROCESSED_DIR - empty folder where will be save processed dataset and augmentation, ssd partition type highly recommended
    - RESULT_DIR - folder wehere weights for neural network will be saved
  - start training sctipt:
  ```bash
  ./train.sh -j X
  ```
  where 'X' is num cpu threads, 6-12 cores recomended
  - wait... :)
  - result will be saved in `RESULT_DIR/final.torch` and `RESULT_DIR/final-half.torch`

  ---
## To inference process 
  - run:

  ```
  ./eval.sh -s csv_result_file -m path_to_model_file -d path_to_folder_with_wav
  ```

  for insnance:
  ```
  ./eval.sh -s result.csv -m exps/exp1/final-half.torch -d /media/ssd/myna-labs/numbers2/test-example/
  ```
  
## Validation results 
> For supervised training process validation on 500 labeled files showed SER: **0.0241**

> For fixmatch training process validation on 500 labeled files showed SER: **0.0023**
  