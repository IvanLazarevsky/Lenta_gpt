import torch
class Checkpoint:
    def __init__(self, chk_path):
        self.chk_path = chk_path
        self.best_path = None
        
    def update(self, data_to_save, index=None, best=False):
        if index is not None:
            path = self.chk_path.format(*index)
        else:
            path = self.chk_path
        
        torch.save(data_to_save, path)
        if best:
            self.best_path = path
    
    def load_best(self):
        return torch.load(self.best_path)
    
    def load(self, index):
        return torch.load(self.chk_path.format(*index) if index is not None else self.chk_path)