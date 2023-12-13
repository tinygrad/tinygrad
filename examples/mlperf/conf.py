from dataclasses import dataclass, field
from typing import Tuple

DATASET_SIZE = 168


@dataclass
class Conf:
  """
  https://github.com/mlcommons/training_policies/blob/master/training_rules.adoc#42-open-division

  unet3d |sgd |global_batch_size              |unconstrained      |global batch size                                |reference --batch_size         | 2
  unet3d |sgd |opt_base_learning_rate         |unconstrained      |base learning rate                               |reference --learning_rate      | 0.8
  unet3d |sgd |opt_momentum                   |unconstrained      |SGD momentum                                     |reference --momentum           | 0.9
  unet3d |sgd |opt_learning_rate_warmup_steps |unconstrained      |number of epochs needed for learning rate warmup |reference --lr_warmup_epochs   | 200
  unet3d |sgd |opt_initial_learning_rate      |unconstrained      |initial learning rate (for LR warm up)           |reference --init_learning_rate | 1e-4
  unet3d |sgd |opt_learning_rate_decay_steps  |unconstrained      |epochs at which the learning rate decays         |reference --lr_decay_epochs    | []
  unet3d |sgd |opt_learning_rate_decay_factor |unconstrained      |factor used for learning rate decay              |reference --lr_decay_factor    | 1.0
  unet3d |sgd |opt_weight_decay               |unconstrained      |L2 weight decay                                  |reference --weight_decay       | 0.0
  unet3d |sgd |training_oversampling          |fixed to reference |training oversampling                            |reference --oversampling       | 0.4
  unet3d |sgd |training_input_shape           |fixed to reference |training input shape                             |reference --input_shape        | (128, 128, 128)
  unet3d |sgd |evaluation_overlap             |fixed to reference |evaluation sliding window overlap                |reference --overlap            | 0.5
  unet3d |sgd |evaluation_input_shape         |fixed to reference |evaluation input shape                           |reference --val_input_shape    | (128, 128, 128)
  unet3d |sgd |data_train_samples             |fixed to reference |number of training samples                       | N/A                           | 168
  unet3d |sgd |data_eval_samples              |fixed to reference |number of evaluation samples                     | N/A                           | 42
  """
  data_dir: str = "./extra/dataset/kits19/data"
  log_dir: str = "/tmp"
  save_ckpt_path: str = ""
  load_ckpt_path: str = ""
  val_split: float = 0.1

  start_epoch: int = 0
  epochs: int = 100
  quality_threshold: float = 0.908
  ga_steps: int = 1
  warmup_step: int = 4
  batch_size: int = 2
  layout: str = "NCDHW"
  # input_shape: Tuple[int, int, int] = (128, 128, 128)
  # val_input_shape: Tuple[int, int, int] = (128, 128, 128)
  input_shape: Tuple[int, int, int] = (64,64,64)
  val_input_shape: Tuple[int, int, int] = (64,64,64)
  seed: int = 0
  num_workers: int = 8
  exec_mode: str = "train"

  benchmark: bool = False
  amp: bool = False
  optimizer: str = "sgd"
  lr: float = 0.08 # 1e-3
  init_lr: float = 1e-4
  lr_warmup_epochs: int = 1 # 200
  lr_decay_epochs: int = field(default_factory=lambda: [1,2])
  lr_decay_factor: float = 0.1
  momentum: float = 0.9
  weight_decay: float = 0.0
  eval_every: int = 10
  start_eval_at: int = 10 # 20
  verbose: bool = True
  normalization: str = "instancenorm"
  activation: str = "relu"

  oversampling: float = 0.4
