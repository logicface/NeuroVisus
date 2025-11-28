import argparse
import yaml
import sys
import os

# 确保能导入 neurovisus (如果 setup.py 没生效，这行是保险)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neurovisus.data.datasets import BioVidPartADataset
from neurovisus.engine.trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description="NeuroVisus Training Launcher")
    parser.add_argument('--config', type=str, default='configs/painformer_default.yaml', help='Path to config file')
    args = parser.parse_args()

    print(f"📖 Loading configuration from: {args.config}")
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print("💿 Loading Dataset...")
    dataset = BioVidPartADataset(
        root_dir=config['data_path'],
        use_subset=config['use_subset'],
        subset_ratio=config['subset_ratio'],
        augment_occlusion=config['augment_occlusion']
    )

    print("🔥 Initializing Trainer...")
    trainer = Trainer(config, dataset)
    
    print("🚀 Starting Training Loop...")
    trainer.run()

if __name__ == '__main__':
    main()