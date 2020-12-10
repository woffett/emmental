"""Emmental LEEP scheduler."""
from typing import Dict, Iterator, List, Tuple, Union

import numpy as np
from torch import Tensor

from emmental.data import EmmentalDataLoader
from emmental.model import EmmentalModel
from emmental.schedulers.scheduler import Scheduler


class LEEPScheduler(Scheduler):
    """Generate batch generator from all dataloaders in sequential order,
    according to a curricula-based LEEP order

    Args:
      fillup: Whether fillup to make all dataloader the same size.
    """

    def __init__(self, fillup: bool = False) -> None:
        """Initialize LEEPScheduler."""
        super().__init__()

        self.fillup = fillup
        self.order = None

    def get_num_batches(self, dataloaders: List[EmmentalDataLoader]) -> int:
        """Get total number of batches per epoch.

        Args:
          dataloaders: List of dataloaders.

        Returns:
          Total number of batches per epoch.
        """
        batch_counts = [len(dataloader) for dataloader in dataloaders]
        if self.fillup:
            batch_counts = [max(batch_counts)] * len(dataloaders)

        for idx in range(len(dataloaders)):
            if dataloaders[idx].n_batches:
                batch_counts[idx] = dataloaders[idx].n_batches

        return sum(batch_counts)

    def leep(self, model, main_dataloader, main_num_labels, aux_task):
        uid = main_dataloader.uid
        aux_num_labels = model.module_pool[f'{aux_task}_pred_head'].out_features
        task_to_label_dict = {aux_task: 'labels'}
        empirical_joint = np.zeros((main_num_labels, aux_num_labels)) # hat_P(y,z)
        z_probs = []
        y_labels = []
        for batch_num, bdict in enumerate(main_dataloader):
            X_bdict, Y_bdict = bdict
            batch_size = len(X_bdict['data'])
            batch_joint = np.zeros((main_num_labels, aux_num_labels))
            (
                uid_bdict, loss_bdict, prob_bdict, gold_bdict
            ) = model.forward(X_bdict[uid], X_bdict, Y_bdict, task_to_label_dict,
                              return_action_outputs=False)
            for i, y_label in enumerate(gold_bdict[aux_task]):
                z_prob = prob_bdict[aux_task][i]
                batch_joint[y_label, :] += z_prob
            batch_joint /= batch_size # normalize for batch
            empirical_joint += batch_joint # add to total empirical
            z_probs.extend(prob_bdict[aux_task])
            y_labels.extend(gold_bdict[aux_task])
            
        empirical_joint /= len(main_dataloader) # normalize across batches
        Pz = np.sum(empirical_joint, axis=0) # sum across y labels
        conditional = empirical_joint
        for i in range(aux_num_labels):
            conditional[:, i] /= Pz[i]

        leep = 0
        for i, ylabel in enumerate(y_labels):
            zprob = z_probs[i]
            cond_prob = conditional[ylabel, :]
            leep += np.log(np.sum(cond_prob * zprob))
        leep /= len(y_labels)
        return leep

    def update_order(self, main_task, model, main_dataloader):
        """Sets curriculum task order using LEEP score
        Args:
          main_task: string describing the task to be optimized for
          model: Emmental model
          main_dataloader: dataloader for main_task
        Returns:
          None
        """

        is_training = model.training
        model.eval()

        # get list of all tasks
        tasks = list(model.task_names)
        main_num_labels = model.module_pool[f'{main_task}_pred_head'].out_features
        
        leeps = []
        for task in tasks:
            # TODO: allow for continuous output support
            # right now only support labels,
            if task == main_task:
                # no need to compute for main_task
                leeps.append(0)
                continue
            num_labels = model.module_pool[f'{task}_pred_head'].out_features
            leeps.append(self.leep(model, main_dataloader, main_num_labels, task))
        
        order_idxs = np.argsort(leeps)[::-1] # in descending order
        self.order = [tasks[idx] for idx in order_idxs if tasks[idx] != main_task]
        self.order += [main_task]

        if is_training:
            model.train()

        return

    def order_to_idxs(self, task_names):
        '''
        Args:
          task_names: a list of task names in the original order

        Returns:
          A list of indices of tasks in task_names reflecting the order of 
          self.order
        '''
        assert self.order is not None, f"Cannot generate order for tasks {task_names} without order set!"
        task2idx = {t: i for i, t in enumerate(task_names)}
        return [task2idx[t] for t in self.order]

    def get_batches(
        self, dataloaders: List[EmmentalDataLoader], model: EmmentalModel = None,
        main_task=None
    ) -> Iterator[
        Tuple[
            List[str],
            Dict[str, Union[Tensor, List[str]]],
            Dict[str, Tensor],
            Dict[str, str],
            str,
            str,
        ]
    ]:
        """Generate batch generator from all dataloaders for one epoch.

        Args:
          dataloaders: List of dataloaders.
          model: The training model, defaults to None.
          main_task: the task for which to construct a LEEP ordering

        Returns:
          A generator of all batches.
        """
        task_to_label_dicts = [
            dataloader.task_to_label_dict for dataloader in dataloaders
        ]
        uid_names = [dataloader.uid for dataloader in dataloaders]
        data_names = [dataloader.data_name for dataloader in dataloaders]
        splits = [dataloader.split for dataloader in dataloaders]
        data_loaders = [iter(dataloader) for dataloader in dataloaders]

        # set task order, if it doesn't exist then generate it
        task_names = [list(d.keys())[0] for d in task_to_label_dicts]
        if self.order is None:
            main_task_idx = [i for i, tn in enumerate(task_names) if tn == main_task][0]
            self.update_order(main_task, model, dataloaders[main_task_idx])

        order_idxs = self.order_to_idxs(task_names)
        # update all lists to reflect new order
        dataloaders = [dataloaders[i] for i in order_idxs]
        task_to_label_dicts = [task_to_label_dicts[i] for i in order_idxs]
        data_names = [data_names[i] for i in order_idxs]
        splits = [splits[i] for i in order_idxs]
        data_loaders = [data_loaders[i] for i in order_idxs]

        # Calc the batch size for each dataloader
        batch_counts = [len(dataloader) for dataloader in dataloaders]
        if self.fillup:
            batch_counts = [max(batch_counts)] * len(dataloaders)

        for idx in range(len(dataloaders)):
            if dataloaders[idx].n_batches:
                batch_counts[idx] = dataloaders[idx].n_batches

        for (
            data_loader_idx,
            (task_to_label_dict, data_name, batch_count, split, uid_name),
        ) in enumerate(
            zip(task_to_label_dicts, data_names, batch_counts, splits, uid_names)
        ):
            for batch_idx in range(batch_count):
                try:
                    X_dict, Y_dict = next(data_loaders[data_loader_idx])
                except StopIteration:
                    data_loaders[data_loader_idx] = iter(dataloaders[data_loader_idx])
                    X_dict, Y_dict = next(data_loaders[data_loader_idx])

                yield X_dict[
                    uid_name
                ], X_dict, Y_dict, task_to_label_dict, data_name, split
