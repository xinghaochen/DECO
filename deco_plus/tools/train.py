# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetr_pytorch)
# ============================================================================

import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse

import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import TASKS


def main(args, ) -> None:
    '''main
    '''
    dist.init_distributed(backend='nccl')
    cfg = YAMLConfig(args.config, resume=args.resume, use_amp=args.amp)
    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    
    if args.test_only:
        solver.val()
    else:
        solver.fit()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='configs/decoplus/decoplus_r18.yml')
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--test-only', action='store_true', default=False,)
    parser.add_argument('--amp', action='store_true', default=False,)

    args = parser.parse_args()

    main(args)
