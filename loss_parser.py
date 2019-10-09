from loss import batch_hard_loss, margin_sample_mining_loss, margin_based_loss, horde_loss, ms_loss, adv_loss, ep_loss, contrastive_loss
from loss import softmax_loss


dml_loss_dict = {
    'trihard': batch_hard_loss,
    'msml': margin_sample_mining_loss,
    'margin': margin_based_loss,
    'ms': ms_loss,
    'ep': ep_loss,
    'contrastive': contrastive_loss,
}


def get_dml_loss(inputs, ids, loss_config):
    loss_name = loss_config['name']
    loss_param = loss_config['param']
    loss_weight = loss_config['weight']

    loss_out = loss_weight * dml_loss_dict[loss_name].loss(inputs, ids, **loss_param)

    return loss_out


def get_cls_loss(inputs, labels, loss_config):
    num_class = loss_config['num_class']
    loss_weight = loss_config['weight']

    loss_out = loss_weight * softmax_loss.loss(inputs, labels, num_class)

    return loss_out
