import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lib.config import cfg
import lib.utils.odious as odious

class SmoothL1Loss(nn.Module):
    def __init__(self,beta=1.0,reduction="mean"):
        super().__init__()
        self.beta=beta
        self.reduction=reduction
    def forward(self,pred,target,weight=None):
        assert pred.size()==target.size() and target.numel()>0
        diff=torch.abs(pred-target)
        loss=torch.where(diff<self.beta,0.5*diff*diff/self.beta,diff-0.5*self.beta)
        if weight is not None:
            loss=loss*weight
        return loss

class WeightedSmoothL1Loss(nn.Module):
    """
    Code-wise Weighted Smooth L1 Loss modified based on fvcore.nn.smooth_l1_loss
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    """
    def __init__(self, beta: float = 1.0 / 9.0, code_weights: list = None):
        """
        Args:
            beta: Scalar float.
                L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)

        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        # code-wise weighting
        if self.code_weights is not None:
            diff = diff * self.code_weights.view(1, 1, -1)

        loss = self.smooth_l1_loss(diff, self.beta)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss

class DiceLoss(nn.Module):
    def __init__(self, ignore_target = -1):
        super().__init__()
        self.ignore_target = ignore_target

    def forward(self, input, target):
        """
        :param input: (N), logit
        :param target: (N), {0, 1}
        :return:
        """
        input = torch.sigmoid(input.view(-1))
        target = target.float().view(-1)
        mask = (target != self.ignore_target).float()
        return 1.0 - (torch.min(input, target) * mask).sum() / torch.clamp((torch.max(input, target) * mask).sum(),
                                                                           min = 1.0)


class SigmoidFocalClassificationLoss(nn.Module):
    """Sigmoid focal cross entropy loss.
      Focal loss down-weights well classified examples and focusses on the hard
      examples. See https://arxiv.org/pdf/1708.02002.pdf for the loss definition.
    """

    def __init__(self, gamma = 2.0, alpha = 0.25):
        """Constructor.
        Args:
            gamma: exponent of the modulating factor (1 - p_t) ^ gamma.
            alpha: optional alpha weighting factor to balance positives vs negatives.
            all_zero_negative: bool. if True, will treat all zero as background.
            else, will treat first label as background. only affect alpha.
        """
        super().__init__()
        self._alpha = alpha
        self._gamma = gamma

    def forward(self,
                prediction_tensor,
                target_tensor,
                weights):
        """Compute loss function.

        Args:
            prediction_tensor: A float tensor of shape [batch_size, num_anchors,
              num_classes] representing the predicted logits for each class
            target_tensor: A float tensor of shape [batch_size, num_anchors,
              num_classes] representing one-hot encoded classification targets
            weights: a float tensor of shape [batch_size, num_anchors]
            class_indices: (Optional) A 1-D integer tensor of class indices.
              If provided, computes loss only for the specified class indices.

        Returns:
          loss: a float tensor of shape [batch_size, num_anchors, num_classes]
            representing the value of the loss function.
        """
        per_entry_cross_ent = (_sigmoid_cross_entropy_with_logits(
                labels = target_tensor, logits = prediction_tensor))
        prediction_probabilities = torch.sigmoid(prediction_tensor)
        p_t = ((target_tensor * prediction_probabilities) +
               ((1 - target_tensor) * (1 - prediction_probabilities)))
        modulating_factor = 1.0
        if self._gamma:
            modulating_factor = torch.pow(1.0 - p_t, self._gamma)
        alpha_weight_factor = 1.0
        if self._alpha is not None:
            alpha_weight_factor = (target_tensor * self._alpha + (1 - target_tensor) * (1 - self._alpha))

        focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor * per_entry_cross_ent)
        return focal_cross_entropy_loss * weights


def _sigmoid_cross_entropy_with_logits(logits, labels):
    # to be compatible with tensorflow, we don't use ignore_idx
    loss = torch.clamp(logits, min = 0) - logits * labels.type_as(logits)
    loss += torch.log1p(torch.exp(-torch.abs(logits)))
    # transpose_param = [0] + [param[-1]] + param[1:-1]
    # logits = logits.permute(*transpose_param)
    # loss_ftor = nn.NLLLoss(reduce=False)
    # loss = loss_ftor(F.logsigmoid(logits), labels)
    return loss

def decode_bbox_target(pred_reg, start_offset, num_head_bin, get_ry_fine=False):
    # recover ry rotation
    ry_bin_l, ry_bin_r = start_offset, start_offset + num_head_bin
    ry_res_l, ry_res_r = ry_bin_r, ry_bin_r + num_head_bin

    assert cfg.TRAIN.RY_WITH_BIN == cfg.TEST.RY_WITH_BIN
    if not cfg.TEST.RY_WITH_BIN: #False
        ry_bin = torch.argmax(pred_reg[:, ry_bin_l: ry_bin_r], dim=1)
        ry_res_norm = torch.gather(pred_reg[:, ry_res_l: ry_res_r], dim=1, index=ry_bin.unsqueeze(dim=1)).squeeze(dim=1)
        if get_ry_fine:
            # divide pi/2 into several bins
            angle_per_class = (np.pi / 2) / num_head_bin
            ry_res = ry_res_norm * (angle_per_class / 2)
            ry = (ry_bin.float() * angle_per_class + angle_per_class / 2) + ry_res - np.pi / 4
        else:
            angle_per_class = (2 * np.pi) / num_head_bin
            ry_res = ry_res_norm * (angle_per_class / 2)

            # bin_center is (0, 30, 60, 90, 120, ..., 270, 300, 330)
            ry = (ry_bin.float() * angle_per_class + ry_res) % (2 * np.pi)
            ry[ry > np.pi] -= 2 * np.pi
    else:
        # print("RY with BIN")
        ry_bin = F.softmax(pred_reg[:, ry_bin_l: ry_bin_r], 1)
        # print(ry_bin[:10, :])
        ry_res_norm = pred_reg[:, ry_res_l: ry_res_r]
        if get_ry_fine:
            # divide pi/2 into several bins
            angle_per_class = (np.pi / 2) / num_head_bin
            ry_res = ry_res_norm * (angle_per_class / 2)
            # ry = (ry_bin.float() * angle_per_class + angle_per_class / 2) + ry_res - np.pi / 4
            ry_bin_ind = torch.arange(num_head_bin).float().to(ry_res_norm.device)
            ry = (ry_bin_ind * angle_per_class + angle_per_class / 2) + ry_res - np.pi / 4
            # [way1]
            # ry = (ry * ry_bin).sum(dim=1)
            # [way2]
            ry_bin_r = ry_bin.clone()
            ry_bin_r[ry < 0] = 0  # [0, pi/4]
            p_rside = ry_bin_r.sum(dim=1, keepdim=True) + 1e-7  # B
            ry_bin_r = ry_bin_r / p_rside

            ry_bin_l = ry_bin.clone()
            ry_bin_l[ry >= 0] = 0  # [-pi/4, 0]
            p_lside = ry_bin_l.sum(dim=1, keepdim=True) + 1e-7
            ry_bin_l = ry_bin_l / p_lside

            # assert 1 - (p_rside + p_lside) < p_lside.new().resize_(p_lside.size()).fill_(1e-4)
            ry_r = ry.clone()
            ry_r[ry_r < 0] = 0
            ry_r = (ry_r * ry_bin_r).sum(dim=1)

            ry_l = ry.clone()
            ry_l[ry_l >= 0] = 0
            ry_l = (ry_l * ry_bin_l).sum(dim=1)

            # flags
            use_r = p_rside.squeeze() >= p_lside.squeeze()
            use_l = p_rside.squeeze() < p_lside.squeeze()
            ry = ry_r * use_r.float() + ry_l * use_l.float()

        else:
            angle_per_class = (2 * np.pi) / num_head_bin
            ry_res = ry_res_norm * (angle_per_class / 2)

            # bin_center is (0, 30, 60, 90, 120, ..., 270, 300, 330)
            # ry = (ry_bin.float() * angle_per_class + ry_res) % (2 * np.pi)
            ry_bin_ind = torch.arange(num_head_bin).float().to(ry_res_norm.device)
            ry = (ry_bin_ind * angle_per_class + ry_res) % (2 * np.pi)

            # [way2] ry [0, 2pi]
            ry_bin_r = ry_bin.clone()
            ry_bin_r[ry > np.pi] = 0  # [0, pi]
            p_rside = ry_bin_r.sum(dim=1, keepdim=True) + 1e-7  # B
            ry_bin_r = ry_bin_r / p_rside

            ry_bin_l = ry_bin.clone()
            ry_bin_l[ry <= np.pi] = 0  # (pi, 2*pi]
            p_lside = ry_bin_l.sum(dim=1, keepdim=True) + 1e-7
            ry_bin_l = ry_bin_l / p_lside

            ry_r = ry.clone()
            ry_r[ry_r > np.pi] = 0
            ry_r = (ry_r * ry_bin_r).sum(dim=1)  # [0, pi]
            # print('ry_r', ry_r.size())

            ry_l = ry.clone()
            ry_l[ry_l <= np.pi] = 0
            ry_l = (ry_l * ry_bin_l).sum(dim=1)  # (pi, 2*pi]
            # print('ry_l', ry_l.size())

            # flags
            use_r = p_rside.squeeze() >= p_lside.squeeze()
            use_l = p_rside.squeeze() < p_lside.squeeze()
            # print('use_r', use_r.size())
            ry = ry_r * use_r.float() + ry_l * use_l.float()
            ry[ry > np.pi] -= 2 * np.pi
    return ry

def get_reg_loss(cls_score, mask_score, pred_reg, reg_label, loc_scope, loc_bin_size, num_head_bin, anchor_size,
                 get_xz_fine = True, get_y_by_bin = False, loc_y_scope = 0.5, loc_y_bin_size = 0.25,
                 get_ry_fine = False,
                 use_cls_score = False, use_mask_score = False,
                 gt_iou_weight = None,
                 use_iou_branch=False,
                 iou_branch_pred=None,
                 use_odiou_branch=False,
                 roi_boxes3d = None
                 ):
    """
    Bin-based 3D bounding boxes regression loss. See https://arxiv.org/abs/1812.04244 for more details.
    :param pred_reg: (N, C)
    :param reg_label: (N, 7) [dx, dy, dz, h, w, l, ry]
    :param loc_scope: constant
    :param loc_bin_size: constant
    :param num_head_bin: constant
    :param anchor_size: (N, 3) or (3)
    :param get_xz_fine:
    :param get_y_by_bin:
    :param loc_y_scope:
    :param loc_y_bin_size:
    :param get_ry_fine:
    :return:
    """
    per_loc_bin_num = int(loc_scope / loc_bin_size) * 2 #[3,0.5,12] [1.5,0.5,6]
    loc_y_bin_num = int(loc_y_scope / loc_y_bin_size) * 2 #[0.5,0.25,4] [0.5,0.25,4]

    reg_loss_dict = { }
    loc_loss = 0

    # xz localization loss
    x_offset_label, y_offset_label, z_offset_label = reg_label[:, 0], reg_label[:, 1], reg_label[:, 2]
    x_shift = torch.clamp(x_offset_label + loc_scope, 0, loc_scope * 2 - 1e-3)
    z_shift = torch.clamp(z_offset_label + loc_scope, 0, loc_scope * 2 - 1e-3)
    x_bin_label = (x_shift / loc_bin_size).floor().long()
    z_bin_label = (z_shift / loc_bin_size).floor().long()

    x_bin_l, x_bin_r = 0, per_loc_bin_num
    z_bin_l, z_bin_r = per_loc_bin_num, per_loc_bin_num * 2
    start_offset = z_bin_r

    loss_x_bin = F.cross_entropy(pred_reg[:, x_bin_l: x_bin_r], x_bin_label)
    loss_z_bin = F.cross_entropy(pred_reg[:, z_bin_l: z_bin_r], z_bin_label)
    reg_loss_dict['loss_x_bin'] = loss_x_bin.item()
    reg_loss_dict['loss_z_bin'] = loss_z_bin.item()
    loc_loss += loss_x_bin + loss_z_bin

    if get_xz_fine:
        x_res_l, x_res_r = per_loc_bin_num * 2, per_loc_bin_num * 3
        z_res_l, z_res_r = per_loc_bin_num * 3, per_loc_bin_num * 4
        start_offset = z_res_r

        x_res_label = x_shift - (x_bin_label.float() * loc_bin_size + loc_bin_size / 2)
        z_res_label = z_shift - (z_bin_label.float() * loc_bin_size + loc_bin_size / 2)
        x_res_norm_label = x_res_label / loc_bin_size
        z_res_norm_label = z_res_label / loc_bin_size

        x_bin_onehot = torch.cuda.FloatTensor(x_bin_label.size(0), per_loc_bin_num).zero_()
        x_bin_onehot.scatter_(1, x_bin_label.view(-1, 1).long(), 1)
        z_bin_onehot = torch.cuda.FloatTensor(z_bin_label.size(0), per_loc_bin_num).zero_()
        z_bin_onehot.scatter_(1, z_bin_label.view(-1, 1).long(), 1)

        loss_x_res = F.smooth_l1_loss((pred_reg[:, x_res_l: x_res_r] * x_bin_onehot).sum(dim = 1), x_res_norm_label)
        loss_z_res = F.smooth_l1_loss((pred_reg[:, z_res_l: z_res_r] * z_bin_onehot).sum(dim = 1), z_res_norm_label)
        reg_loss_dict['loss_x_res'] = loss_x_res.item()
        reg_loss_dict['loss_z_res'] = loss_z_res.item()
        loc_loss += loss_x_res + loss_z_res

    # y localization loss
    if get_y_by_bin: #False
        y_bin_l, y_bin_r = start_offset, start_offset + loc_y_bin_num
        y_res_l, y_res_r = y_bin_r, y_bin_r + loc_y_bin_num
        start_offset = y_res_r

        y_shift = torch.clamp(y_offset_label + loc_y_scope, 0, loc_y_scope * 2 - 1e-3)
        y_bin_label = (y_shift / loc_y_bin_size).floor().long()
        y_res_label = y_shift - (y_bin_label.float() * loc_y_bin_size + loc_y_bin_size / 2)
        y_res_norm_label = y_res_label / loc_y_bin_size

        y_bin_onehot = torch.cuda.FloatTensor(y_bin_label.size(0), loc_y_bin_num).zero_()
        y_bin_onehot.scatter_(1, y_bin_label.view(-1, 1).long(), 1)

        loss_y_bin = F.cross_entropy(pred_reg[:, y_bin_l: y_bin_r], y_bin_label)
        loss_y_res = F.smooth_l1_loss((pred_reg[:, y_res_l: y_res_r] * y_bin_onehot).sum(dim = 1), y_res_norm_label)

        reg_loss_dict['loss_y_bin'] = loss_y_bin.item()
        reg_loss_dict['loss_y_res'] = loss_y_res.item()

        loc_loss += loss_y_bin + loss_y_res
    else:
        y_offset_l, y_offset_r = start_offset, start_offset + 1
        start_offset = y_offset_r

        loss_y_offset = F.smooth_l1_loss(pred_reg[:, y_offset_l: y_offset_r].sum(dim = 1), y_offset_label)
        reg_loss_dict['loss_y_offset'] = loss_y_offset.item()
        loc_loss += loss_y_offset

    # angle loss
    ry_bin_l, ry_bin_r = start_offset, start_offset + num_head_bin
    ry_res_l, ry_res_r = ry_bin_r, ry_bin_r + num_head_bin

    ry_label = reg_label[:, 6]

    if get_ry_fine:
        # divide pi/2 into several bins (For RCNN, num_head_bin = 9)
        angle_per_class = (np.pi / 2) / num_head_bin

        ry_label = ry_label % (2 * np.pi)  # 0 ~ 2pi
        opposite_flag = (ry_label > np.pi * 0.5) & (ry_label < np.pi * 1.5)
        ry_label[opposite_flag] = (ry_label[opposite_flag] + np.pi) % (2 * np.pi)  # (0 ~ pi/2, 3pi/2 ~ 2pi)
        shift_angle = (ry_label + np.pi * 0.5) % (2 * np.pi)  # (0 ~ pi)

        shift_angle = torch.clamp(shift_angle - np.pi * 0.25, min = 1e-3, max = np.pi * 0.5 - 1e-3)  # (0, pi/2)

        # bin center is (5, 10, 15, ..., 85)
        ry_bin_label = (shift_angle / angle_per_class).floor().long()
        ry_res_label = shift_angle - (ry_bin_label.float() * angle_per_class + angle_per_class / 2)
        ry_res_norm_label = ry_res_label / (angle_per_class / 2)

    else:
        # divide 2pi into several bins (For RPN, num_head_bin = 12)
        angle_per_class = (2 * np.pi) / num_head_bin
        heading_angle = ry_label % (2 * np.pi)  # 0 ~ 2pi

        shift_angle = (heading_angle + angle_per_class / 2) % (2 * np.pi)
        ry_bin_label = (shift_angle / angle_per_class).floor().long()
        ry_res_label = shift_angle - (ry_bin_label.float() * angle_per_class + angle_per_class / 2)
        ry_res_norm_label = ry_res_label / (angle_per_class / 2)

    ry_bin_onehot = torch.cuda.FloatTensor(ry_bin_label.size(0), num_head_bin).zero_()
    ry_bin_onehot.scatter_(1, ry_bin_label.view(-1, 1).long(), 1) #one-hot??????
    loss_ry_bin = F.cross_entropy(pred_reg[:, ry_bin_l:ry_bin_r], ry_bin_label)
    loss_ry_res = F.smooth_l1_loss((pred_reg[:, ry_res_l: ry_res_r] * ry_bin_onehot).sum(dim = 1), ry_res_norm_label)
    reg_loss_dict['loss_ry_bin'] = loss_ry_bin.item()
    reg_loss_dict['loss_ry_res'] = loss_ry_res.item()
    angle_loss = loss_ry_bin + loss_ry_res

    # size loss
    size_res_l, size_res_r = ry_res_r, ry_res_r + 3
    assert pred_reg.shape[1] == size_res_r, '%d vs %d' % (pred_reg.shape[1], size_res_r)

    size_res_norm_label = (reg_label[:, 3:6] - anchor_size) / anchor_size
    size_res_norm = pred_reg[:, size_res_l:size_res_r]
    size_loss = F.smooth_l1_loss(size_res_norm, size_res_norm_label)

    # iou loss
    pred_x = (pred_reg[:, x_res_l: x_res_r] * x_bin_onehot).sum(dim = 1) * loc_bin_size
    pred_y = pred_reg[:, y_offset_l: y_offset_r].sum(dim = 1)
    pred_z = (pred_reg[:, z_res_l: z_res_r] * z_bin_onehot).sum(dim = 1) * loc_bin_size
    pred_size = size_res_norm * anchor_size + anchor_size  # hwl(yzx)

    tar_x, tar_y, tar_z = x_res_label, y_offset_label, z_res_label
    tar_size = reg_label[:, 3:6]

    insect_x = torch.max(torch.min((pred_x + pred_size[:, 2] / 2), (tar_x + tar_size[:, 2] / 2)) - torch.max(
            (pred_x - pred_size[:, 2] / 2), (tar_x - tar_size[:, 2] / 2)),
                         pred_x.new().resize_(pred_x.shape).fill_(1e-3))
    insect_y = torch.max(torch.min((pred_y + pred_size[:, 0] / 2), (tar_y + tar_size[:, 0] / 2)) - torch.max(
            (pred_y - pred_size[:, 0] / 2), (tar_y - tar_size[:, 0] / 2)),
                         pred_x.new().resize_(pred_x.shape).fill_(1e-3))
    insect_z = torch.max(torch.min((pred_z + pred_size[:, 1] / 2), (tar_z + tar_size[:, 1] / 2)) - torch.max(
            (pred_z - pred_size[:, 1] / 2), (tar_z - tar_size[:, 1] / 2)),
                         pred_x.new().resize_(pred_x.shape).fill_(1e-3))


    if cfg.TRAIN.IOU_LOSS_TYPE == 'raw':
        # print('USE RAW LOSS')
        insect_area = insect_x * insect_y * insect_z
        pred_area = torch.max(pred_size[:, 0] * pred_size[:, 1] * pred_size[:, 2],
                              pred_size.new().resize_(pred_size[:, 2].shape).fill_(1e-3))
        tar_area = tar_size[:, 0] * tar_size[:, 1] * tar_size[:, 2]
        iou_tmp = insect_area / (pred_area + tar_area - insect_area)

        if use_iou_branch:
            iou_branch_pred_flat = iou_branch_pred.view(-1)
            iou_branch_pred_flat = torch.clamp(iou_branch_pred_flat, 0.0001, 0.9999)
            iou_tmp_taget = torch.clamp(iou_tmp, 0.0001, 0.9999)
            iou_branch_loss = -(iou_tmp_taget.detach() * torch.log(iou_branch_pred_flat) + (
                        1 - iou_tmp_taget.detach()) * torch.log(1 - iou_branch_pred_flat))
            reg_loss_dict['iou_branch_loss'] = iou_branch_loss.mean()

        if use_cls_score:
            iou_tmp = cls_score * iou_tmp

        if use_mask_score:
            # iou_tmp = mask_score * iou_tmp
            iou_tmp = iou_tmp
        iou_tmp = torch.max(iou_tmp, iou_tmp.new().resize_(iou_tmp.shape).fill_(1e-4))
        iou_loss = -torch.log(iou_tmp)
        iou_loss = iou_loss.mean()

    elif cfg.TRAIN.IOU_LOSS_TYPE == 'cls_mask_with_bin':
        #print('cfg.TRAIN.IOU_LOSS_TYPE')
        pred_x_bin = F.softmax(pred_reg[:, x_bin_l: x_bin_r], 1) #???????????????????????????
        pred_z_bin = F.softmax(pred_reg[:, z_bin_l: z_bin_r], 1)
        #print(x_bin_l,x_bin_r,z_bin_l,z_bin_r) #0-12,12-24 / 0-6,6-12

        #
        xz_bin_ind = torch.arange(per_loc_bin_num).float()
        xz_bin_center = xz_bin_ind * loc_bin_size + loc_bin_size / 2 - loc_scope # num_bin
        xz_bin_center = xz_bin_center.to(pred_x_bin.device)

        #
        pred_x_reg = pred_reg[:, x_res_l: x_res_r] * loc_bin_size # N x num_bin
        pred_z_reg = pred_reg[:, z_res_l: z_res_r] * loc_bin_size
        #print(x_res_l,x_res_r,z_res_l,z_res_r) #24-36,36-48 / 12-18,18-24
        #
        pred_x_abs = xz_bin_center + pred_x_reg
        pred_z_abs = xz_bin_center + pred_z_reg
        pred_x = (pred_x_abs * pred_x_bin).sum(dim=1)
        pred_z = (pred_z_abs * pred_z_bin).sum(dim=1)
        pred_y = pred_reg[:, y_offset_l: y_offset_r].sum(dim=1) # N
        #print(y_offset_l,y_offset_r) #48-49 / 24-25
        pred_size = size_res_norm * anchor_size + anchor_size # hwl(yzx)
        #print(size_res_l,size_res_r) #73-76 / 43-46
        pred_ry = decode_bbox_target(pred_reg=pred_reg,start_offset=ry_bin_l,num_head_bin=num_head_bin,get_ry_fine=get_ry_fine)
        #print(ry_bin_l,ry_bin_r,ry_res_l,ry_res_r) #49-61,61-73 / 25-34,34-43
        pred_iou_reg=torch.cat((pred_x.view(-1,1),pred_y.view(-1,1),pred_z.view(-1,1),pred_size[:,1].view(-1,1),pred_size[:,2].view(-1,1),pred_size[:,0].view(-1,1),pred_ry.view(-1,1)),dim=1)
        #
        tar_x, tar_y, tar_z = x_res_label, y_offset_label, z_res_label
        tar_x = xz_bin_center[x_bin_label] + tar_x
        tar_z = xz_bin_center[z_bin_label] + tar_z
        tar_size = reg_label[:, 3:6]
        tar_ry = reg_label[:,6]
        #
        iou_reg=torch.cat((tar_x.view(-1,1),tar_y.view(-1,1),tar_z.view(-1,1),tar_size[:,1].view(-1,1),tar_size[:,2].view(-1,1),tar_size[:,0].view(-1,1),tar_ry.view(-1,1)),dim=1)

        insect_x = torch.max(torch.min((pred_x + pred_size[:, 2]/2), (tar_x + tar_size[:, 2]/2)) - torch.max((pred_x - pred_size[:, 2]/2), (tar_x - tar_size[:, 2]/2)), pred_x.new().resize_(pred_x.shape).fill_(1e-3))
        insect_y = torch.max(torch.min((pred_y + pred_size[:, 0]/2), (tar_y + tar_size[:, 0]/2)) - torch.max((pred_y - pred_size[:, 0]/2), (tar_y - tar_size[:, 0]/2)), pred_x.new().resize_(pred_x.shape).fill_(1e-3))
        insect_z = torch.max(torch.min((pred_z + pred_size[:, 1]/2), (tar_z + tar_size[:, 1]/2)) - torch.max((pred_z - pred_size[:, 1]/2), (tar_z - tar_size[:, 1]/2)), pred_x.new().resize_(pred_x.shape).fill_(1e-3))

        insect_area = insect_x * insect_y * insect_z
        pred_area = torch.max(pred_size[:, 0] * pred_size[:, 1] * pred_size[:, 2], pred_size.new().resize_(pred_size[:, 2].shape).fill_(1e-3))
        tar_area = tar_size[:, 0] * tar_size[:, 1] * tar_size[:, 2]
        iou_tmp = insect_area/(pred_area+tar_area-insect_area)

        if use_iou_branch:
            iou_branch_pred_flat = iou_branch_pred.view(-1)
            iou_branch_pred_flat = torch.clamp(iou_branch_pred_flat, 0.0001, 0.9999)
            iou_tmp_target = torch.clamp(iou_tmp, 0.0001, 0.9999)
            #binary cross entropy losses
            iou_pred_loss = -(iou_tmp_target.detach() * torch.log(iou_branch_pred_flat) + (
                        1 - iou_tmp_target.detach()) * torch.log(1 - iou_branch_pred_flat))
            reg_loss_dict['iou_branch_loss'] = iou_pred_loss.mean()

        if use_cls_score:
            iou_tmp = cls_score * iou_tmp

        if use_mask_score:
            # iou_tmp = mask_score * iou_tmp
            iou_tmp = iou_tmp
        iou_tmp = torch.max(iou_tmp, iou_tmp.new().resize_(iou_tmp.shape).fill_(1e-4))
        iou_loss = -torch.log(iou_tmp)
        iou_loss = iou_loss.mean()

    #odious loss
    if use_odiou_branch:
        #pred_iou_reg[:,6]+=roi_boxes3d[:,6]
        odious_3d_loss=odious.odiou_3D()
        loss_odious=odious_3d_loss(iou_reg,pred_iou_reg,weights=gt_iou_weight,batch_size=0)
        reg_loss_dict['loss_odious']=loss_odious

    #Total regression loss
    reg_loss_dict['loss_loc'] = loc_loss
    reg_loss_dict['loss_angle'] = angle_loss
    reg_loss_dict['loss_size'] = size_loss
    reg_loss_dict['loss_iou'] = iou_loss

    return loc_loss, angle_loss, size_loss, iou_loss, reg_loss_dict
