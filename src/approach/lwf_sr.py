from .incremental_learning import Inc_Learning_Appr
from .lwf import Appr as LwF

class Appr(LwF):
    """LwF with Selective Review (LwF-SR) approach"""
    def __init__(self, model, device, **kwargs):
        super().__init__(model, device, **kwargs)
        
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = LwF.extra_parser(args)
        # 添加LwF-SR特有的参数
        parser.add_argument('--select-threshold', default=0.5, type=float,
                          help='Threshold for sample selection')
        return parser, args
        
    def train(self, t, trn_loader, val_loader):
        # 实现LwF-SR的训练逻辑
        pass 