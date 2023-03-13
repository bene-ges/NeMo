# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
This script contains an example on how to run itn inference with the SpellcheckingAsrCustomizationModel.

The inference works on a raw file (no labels required).
Each line of the input file represents a single example for inference.
    Specify inference.from_file and inference.batch_size parameters.

USAGE Example:
1. Train a model, or use a pretrained checkpoint.
2. Run:
    export TOKENIZERS_PARALLELISM=false
    python ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/spellchecking_asr_customization_train.py \
      pretrained_model=./training.nemo \
      inference.from_file=./input.txt \
      inference.out_file=./output.txt \
      model.max_sequence_len=1024 #\
      inference.batch_size=128

This script uses the `/examples/nlp/spellchecking_asr_customization/conf/spellchecking_asr_customization_config.yaml`
config file by default. The other option is to set another config file via command
line arguments by `--config-name=CONFIG_FILE_PATH'.
"""


import os

from helpers import MODEL, instantiate_model_and_trainer
from omegaconf import DictConfig, OmegaConf

from nemo.core.config import hydra_runner
from nemo.utils import logging


@hydra_runner(config_path="conf", config_name="spellchecking_asr_customization_config")
def main(cfg: DictConfig) -> None:
    logging.debug(f'Config Params: {OmegaConf.to_yaml(cfg)}')

    if cfg.pretrained_model is None:
        raise ValueError("A pre-trained model should be provided.")
    _, model = instantiate_model_and_trainer(cfg, MODEL, False)

    if cfg.model.max_sequence_len != model.max_sequence_len:
        model.max_sequence_len = cfg.model.max_sequence_len
        model.builder._max_seq_length = cfg.model.max_sequence_len
    input_filenames = []
    output_filenames = []

    if "from_filelist" in cfg.inference and "output_folder" in cfg.inference:
        filelist_file = cfg.inference.from_filelist
        output_folder = cfg.inference.output_folder
        with open(filelist_file, "r", encoding="utf-8") as f:
            for line in f:
                path = line.strip()
                input_filenames.append(path)
                folder, name = os.path.split(path)
                output_filenames.append(os.path.join(output_folder, name))
    else:
        text_file = cfg.inference.from_file
        logging.info(f"Running inference on {text_file}...")
        if not os.path.exists(text_file):
            raise ValueError(f"{text_file} not found.")
        input_filenames.append(text_file)
        output_filenames.append(cfg.inference.out_file) 

    batch_size = cfg.inference.get("batch_size", 8)

    for input_filename, output_filename in zip(input_filenames, output_filenames):

        with open(input_filename, "r", encoding="utf-8") as f:
            lines = f.readlines()

        batch, all_preds = [], []
        for i, line in enumerate(lines):
            batch.append(line.strip())
            if len(batch) == batch_size or i == len(lines) - 1:
                outputs = model._infer(batch)
                for x in outputs:
                    all_preds.append(x)
                batch = []
        if len(all_preds) != len(lines):
            raise ValueError(
                "number of input lines and predictions is different: predictions="
                + str(len(all_preds))
                + "; lines="
                + str(len(lines))
            )
        with open(output_filename, "w", encoding="utf-8") as f_out:
            f_out.write("\n".join(all_preds))
        logging.info(f"Predictions saved to {output_filename}.")


if __name__ == "__main__":
    main()
