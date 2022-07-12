"""
Taken from https://raw.githubusercontent.com/opendilab/DI-engine/main/ding/reward_model/guided_cost_reward_model.py
"""

from typing import List, Dict, Any, Tuple, Union, Optional
from easydict import EasyDict

import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Independent, Normal
import copy

def default_collate(batch: Sequence,
                    cat_1dim: bool = True,
                    ignore_prefix: list = ['collate_ignore']) -> Union[torch.Tensor, Mapping, Sequence]:
    """
    Overview:
        Put each data field into a tensor with outer dimension batch size.
    Example:
        >>> # a list with B tensors shaped (m, n) -->> a tensor shaped (B, m, n)
        >>> a = [torch.zeros(2,3) for _ in range(4)]
        >>> default_collate(a).shape
        torch.Size([4, 2, 3])
        >>>
        >>> # a list with B lists, each list contains m elements -->> a list of m tensors, each with shape (B, )
        >>> a = [[0 for __ in range(3)] for _ in range(4)]
        >>> default_collate(a)
        [tensor([0, 0, 0, 0]), tensor([0, 0, 0, 0]), tensor([0, 0, 0, 0])]
        >>>
        >>> # a list with B dicts, whose values are tensors shaped :math:`(m, n)` -->>
        >>> # a dict whose values are tensors with shape :math:`(B, m, n)`
        >>> a = [{i: torch.zeros(i,i+1) for i in range(2, 4)} for _ in range(4)]
        >>> print(a[0][2].shape, a[0][3].shape)
        torch.Size([2, 3]) torch.Size([3, 4])
        >>> b = default_collate(a)
        >>> print(b[2].shape, b[3].shape)
        torch.Size([4, 2, 3]) torch.Size([4, 3, 4])
    Arguments:
        - batch (:obj:`Sequence`): a data sequence, whose length is batch size, whose element is one piece of data
    Returns:
        - ret (:obj:`Union[torch.Tensor, Mapping, Sequence]`): the collated data, with batch size into each data field.\
            the return dtype depends on the original element dtype, can be [torch.Tensor, Mapping, Sequence].
    """
    elem = batch[0]

    elem_type = type(elem)
    if isinstance(batch, ttorch.Tensor):
        return batch.json()
    if isinstance(elem, torch.Tensor):
        out = None
        if torch_ge_131() and torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, directly concatenate into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        if elem.shape == (1, ) and cat_1dim:
            # reshape (B, 1) -> (B)
            return torch.cat(batch, 0, out=out)
            # return torch.stack(batch, 0, out=out)
        else:
            return torch.stack(batch, 0, out=out)
    elif isinstance(elem, ttorch.Tensor):
        ret = ttorch.stack(batch).json()
        for k in ret:
            if len(ret[k].shape) == 2 and ret[k].shape[1] == 1:  # reshape (B, 1) -> (B)
                ret[k] = ret[k].squeeze(1)
        return ret
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))
            return default_collate([torch.as_tensor(b) for b in batch], cat_1dim=cat_1dim)
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float32)
    elif isinstance(elem, int_classes):
        dtype = torch.bool if isinstance(elem, bool) else torch.int64
        return torch.tensor(batch, dtype=dtype)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        ret = {}
        for key in elem:
            if any([key.startswith(t) for t in ignore_prefix]):
                ret[key] = [d[key] for d in batch]
            else:
                ret[key] = default_collate([d[key] for d in batch], cat_1dim=cat_1dim)
        return ret
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples, cat_1dim=cat_1dim) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples, cat_1dim=cat_1dim) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


class GuidedCostNN(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size=128,
        output_size=1,
    ):
        super(GuidedCostNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.net(x)


# @REWARD_MODEL_REGISTRY.register('guided_cost')
class GuidedCostRewardModel():
    r"""
    Overview:
        Policy class of Guided cost algorithm.

        https://arxiv.org/pdf/1603.00448.pdf

    """

    config = dict(
        type='guided_cost',
        learning_rate=1e-3,
        action_shape=1,
        continuous=True,
        batch_size=64,
        hidden_size=128,
        update_per_collect=100,
        log_every_n_train=50,
        store_model_every_n_train=100,
    )

    def __init__(self, config: EasyDict, device: str, tb_logger: 'SummaryWriter') -> None:  # noqa
        super(GuidedCostRewardModel, self).__init__()
        self.cfg = config
        self.action_shape = self.cfg.action_shape
        assert device == "cpu" or device.startswith("cuda")
        self.device = device
        self.tb_logger = tb_logger
        self.reward_model = GuidedCostNN(config.input_size, config.hidden_size)
        self.reward_model.to(self.device)
        self.opt = optim.Adam(self.reward_model.parameters(), lr=config.learning_rate)

    def train(self, expert_demo: torch.Tensor, samp: torch.Tensor, iter, step):
        device_0 = expert_demo[0]['obs'].device
        device_1 = samp[0]['obs'].device
        for i in range(len(expert_demo)):
            expert_demo[i]['prob'] = torch.FloatTensor([1]).to(device_0)
        if self.cfg.continuous:
            for i in range(len(samp)):
                (mu, sigma) = samp[i]['logit']
                dist = Independent(Normal(mu, sigma), 1)
                next_action = samp[i]['action']
                log_prob = dist.log_prob(next_action)
                samp[i]['prob'] = torch.exp(log_prob).unsqueeze(0).to(device_1)
        else:
            for i in range(len(samp)):
                probs = F.softmax(samp[i]['logit'], dim=-1)
                prob = probs[samp[i]['action']]
                samp[i]['prob'] = prob.to(device_1)
        # Mix the expert data and sample data to train the reward model.
        samp.extend(expert_demo)
        expert_demo = default_collate(expert_demo)
        samp = default_collate(samp)
        cost_demo = self.reward_model(
            torch.cat([expert_demo['obs'], expert_demo['action'].float().reshape(-1, self.action_shape)], dim=-1)
        )
        cost_samp = self.reward_model(
            torch.cat([samp['obs'], samp['action'].float().reshape(-1, self.action_shape)], dim=-1)
        )

        prob = samp['prob'].unsqueeze(-1)
        loss_IOC = torch.mean(cost_demo) + \
            torch.log(torch.mean(torch.exp(-cost_samp)/(prob+1e-7)))
        # UPDATING THE COST FUNCTION
        self.opt.zero_grad()
        loss_IOC.backward()
        self.opt.step()
        if iter % self.cfg.log_every_n_train == 0:
            self.tb_logger.add_scalar('reward_model/loss_iter', loss_IOC, iter)
            self.tb_logger.add_scalar('reward_model/loss_step', loss_IOC, step)

    def estimate(self, data: list) -> List[Dict]:
        # NOTE: this estimate method of gcl alg. is a little different from the one in other irl alg.,
        # because its deepcopy is operated before learner train loop.
        train_data_augmented = data
        for i in range(len(train_data_augmented)):
            with torch.no_grad():
                reward = self.reward_model(
                    torch.cat([train_data_augmented[i]['obs'], train_data_augmented[i]['action'].float()]).unsqueeze(0)
                ).squeeze(0)
                train_data_augmented[i]['reward'] = -reward

        return train_data_augmented

    def collect_data(self, data) -> None:
        """
        Overview:
            Collecting training data, not implemented if reward model (i.e. online_net) is only trained ones, \
                if online_net is trained continuously, there should be some implementations in collect_data method
        """
        # if online_net is trained continuously, there should be some implementations in collect_data method
        pass

    def clear_data(self):
        """
        Overview:
            Collecting clearing data, not implemented if reward model (i.e. online_net) is only trained ones, \
                if online_net is trained continuously, there should be some implementations in clear_data method
        """
        # if online_net is trained continuously, there should be some implementations in clear_data method
        pass

    def state_dict_reward_model(self) -> Dict[str, Any]:
        return {
            'model': self.reward_model.state_dict(),
            'optimizer': self.opt.state_dict(),
        }

    def load_state_dict_reward_model(self, state_dict: Dict[str, Any]) -> None:
        self.reward_model.load_state_dict(state_dict['model'])
        self.opt.load_state_dict(state_dict['optimizer'])
