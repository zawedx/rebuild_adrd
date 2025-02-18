from .ml_frame import with_local_info
import wandb

class MLLogger:
    @with_local_info
    def __init__(self,
                 use_wandb=None, local_rank=None, project_name=None, model_name=None):
        assert(local_rank == 0)
        if use_wandb:
            wandb.init(
                project=project_name,
                name=model_name,
                config={
                     # log the config info you want to track
                },
		        mode="online"
            )
            wandb.run.log_code(".")
            self.wandb = wandb
        else:
            self.wandb = None

    @with_local_info
    def log(self, msg, **kwargs):
        print(msg)

    @with_local_info
    def info(self, log_info,
             current_metric=None, epoch=None, tgt_modalities=None, local_rank=None, training_state=None):
        if local_rank != 0:
            return
        training_state_str = "Train" if training_state == "train" else "Validation"
        self.wandb.log({f"{training_state_str} {log_info} {list(tgt_modalities)[i]}": current_metric[i][log_info]  for i in range(len(tgt_modalities))}, step=epoch)