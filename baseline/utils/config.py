import json
import torch

from baseline.utils.normalizer import StateNormalizer, ObservationNormalizer


class Config:
    
    def __init__(
            self,
            args_file,
            guidance_fn=None, # 建议给个默认值，增加健壮性
            **kwargs          # [关键修改] 接收 render_save_dir 等所有额外参数
    ):
        # 1. 先加载 JSON 文件中的配置
        with open(args_file, 'r') as f:
            args_dict = json.load(f)
            
        for key, value in args_dict.items():
            setattr(self, key, value)

        # 2. [新增] 处理 Hydra 命令行传入的额外参数
        # 这一步会将 render_save_dir=... 写入到 self.render_save_dir
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Older StylePlanner training exports (including A3.8) omitted the
        # DataProcessor-only route polyline length even though the originating
        # baseline used route_len == lane_len == 20. Recover that exact input
        # contract without mutating the archived args.json or checkpoint.
        self.config_compatibility_fallbacks = {}
        if not hasattr(self, 'route_len'):
            if not hasattr(self, 'lane_len'):
                raise AttributeError(
                    "args.json is missing both route_len and lane_len; "
                    "the closed-loop input geometry cannot be reconstructed."
                )
            self.route_len = int(self.lane_len)
            self.config_compatibility_fallbacks['route_len'] = {
                'source': 'lane_len',
                'value': self.route_len,
            }
            print(
                "[ConfigCompat] args.json missing route_len; "
                f"recovered route_len={self.route_len} from lane_len."
            )
        if int(self.route_len) <= 0:
            raise ValueError(f"route_len must be positive, got {self.route_len!r}")
            
        # 3. [新增] 兜底逻辑：如果既没传参也没在JSON里，默认为 None
        if not hasattr(self, 'render_save_dir'):
            self.render_save_dir = None

        # 初始化状态归一化器，使用配置文件中的均值和标准差参数
        # (注意：self.state_normalizer 在 setattr 时是 dict，这里被覆盖为 Object)
        self.state_normalizer = StateNormalizer(self.state_normalizer['mean'], self.state_normalizer['std'])

        # 初始化观测归一化器，将配置文件中的均值和标准差转换为张量格式
        self.observation_normalizer = ObservationNormalizer({
            k: {
                'mean': torch.as_tensor(v['mean']),
                'std': torch.as_tensor(v['std'])
            } for k, v in self.observation_normalizer.items()
        })
        
        self.guidance_fn = guidance_fn
