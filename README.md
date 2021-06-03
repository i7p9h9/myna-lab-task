## Hardware requirements
  - GPU with 11Gb memory is necessary
  - 64Gb RAM memory at least
  - 120Gb free space is required (ssd type partition is recommended)

***

## Start with docker
  - `git clone git@github.com:i7p9h9/myna-lab-task.git`
  - cd mina-lab-task/
  - prepare docker environment:
  ```bash
  ./local/prepare_for_docker.sh
  ```

***

## To run training process 
  - edit `path.sh`
    - DATASETS_DIR - path to folder with trainig data, next structure expected:
    ```bash
    .
    ├── test-example
    ├── test-example.csv
    ├── train.csv
    └── train
    ```
    - PROCESSED_DIR - empty folder where processed dataset and augmentation will be saved, ssd partition type highly recommended
    - RESULT_DIR - folder where weights for neural network will be saved
  - start training script:
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

  for instance:
  ```
  ./eval.sh -s result.csv -m exps/exp1/final-half.torch -d /media/ssd/myna-labs/numbers2/test-example/
  ```

---
## Validation results 
  1. For supervised training process validation on 500 labeled files showed CER: **0.0040**
  2. For fixmatch (semi-supervised) training process validation on 500 labeled files showed CER: **0.0017**
  