
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class nnUNetTrainer_Student(nnUNetTrainer):
    """Minimal trainer for student phase - only sets number of epochs"""
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device=None):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        
        # Override number of epochs
        self.num_epochs = 800
        self.initial_lr = 0.005
        
    def on_train_epoch_start(self):
        """Log training progress"""
        super().on_train_epoch_start()
        if self.current_epoch % 1 == 0:
            self.print_to_log_file(f"Student Training - Epoch {self.current_epoch}/{self.num_epochs}")
