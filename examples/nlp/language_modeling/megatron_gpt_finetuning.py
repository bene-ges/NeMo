# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from nemo.utils.app_state import AppState
from pathlib import Path
from nemo.collections.nlp.modules.common.megatron.megatron_utils import compute_model_parallel_rank
from omegaconf.omegaconf import OmegaConf
from pytorch_lightning import Trainer

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.nlp_overrides import (
    NLPCheckpointConnector,
    NLPDDPPlugin,
    NLPNativeMixedPrecisionPlugin,
    NLPPrecisionPlugin,
    NLPSaveRestoreConnector,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="conf", config_name="megatron_gpt_config")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    trainer = None
    if cfg.trainer.precision == 32:
        trainer = Trainer(
            plugins=[NLPDDPPlugin(num_nodes=cfg.trainer.num_nodes), NLPNativeMixedPrecisionPlugin()], **cfg.trainer
        )
    else:
        trainer = Trainer(plugins=[NLPDDPPlugin(num_nodes=cfg.trainer.num_nodes), NLPPrecisionPlugin()], **cfg.trainer)

    app_state = AppState()
    app_state.model_parallel_size = cfg.model.tensor_model_parallel_size
    app_state.model_parallel_rank = compute_model_parallel_rank(trainer.local_rank, app_state.model_parallel_size)



    model = MegatronGPTModel.restore_from(
        '/NeMo/datasets/models/megatron_gpt_rank2.nemo', trainer=trainer
    )

    model.cfg.data.data_prefix = cfg.model.data.data_prefix

    model.cfg.data.splits_string = cfg.model.data.splits_string

    model.cfg.data.seq_length = 1536

    model.cfg.tensor_model_parallel_size = 1

    trainer.fit(model)

    if cfg.model.get('nemo_file_path', None) is not None:
        model.save_to(cfg.model.nemo_file_path)


if __name__ == '__main__':
    main()
