from .incremental_learning import Inc_Learning_Appr
from .lwf import Appr as LwF

class Appr(LwF):
    """LwF with Review (LwF-R) approach"""
    def __init__(self, model, device, **kwargs):
        super().__init__(model, device, **kwargs)
        
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = LwF.extra_parser(args)
        # 添加LwF-R特有的参数
        parser.add_argument('--review-epochs', default=1, type=int, 
                          help='Number of review epochs')
        return parser, args
        
    def train(self, t, trn_loader, val_loader):
        # 实现LwF-R的训练逻辑
        pass 