import torch
import torch.nn.functional as F

def dual_temperature_loss_func(
    rep: torch.Tensor,
    true_y: torch.Tensor,
    y_mask: torch.Tensor,
    temperature=0.1,
    dt_m=10,
) -> torch.Tensor:
    """
    rep: [b, dim] all representations,
    true_y: [b] with values in [0, 1] indicating the true positive label
    y_mask: [b] with values in [0, 1] indicating the mask for the valid labels
    """
    # query: [b, c]
    # key: [b, c]
    # pos_map: [b, b]
    # eye_map: [b, b] with diagonal 0 and others 1
    # sim = query @ key.T
    # assert(true_y.sum() > 0)
    # print(f'true_y = {true_y}')
    # print(f'y_mask = {y_mask}')
    # assert(not torch.isnan(rep).any())
    # assert(not torch.isnan(true_y).any())
    # assert(not torch.isnan(y_mask).any())
    b = rep.shape[0]
    # print("max element in rep", torch.max(rep))
    norm_rep = F.normalize(rep, p=2, dim=1, eps=1e-12)
    # print("max element in norm_rep", torch.max(norm_rep))

    sim_mat = norm_rep @ norm_rep.t() * (1 - torch.eye(b, device=rep.device))
    sim_mat = sim_mat - torch.max(sim_mat)
    # print("max element in sim_mat", torch.max(sim_mat))
    # assert(not torch.isinf(sim_mat).any())
    # set diagonal to -inf
    y_mask = y_mask.float()
    # assert(torch.max(y_mask))
    y_mask_map = y_mask.unsqueeze(1) @ y_mask.unsqueeze(0) * (1 - torch.eye(b, device=rep.device))
    # print(f'y_mask_map = \n{y_mask_map}')
    # sim_mat = sim_mat + (1 - y_mask_map) * -1e9
    # assert(not torch.isinf(sim_mat).any())
    y_row = true_y.unsqueeze(1).repeat(1, b)
    y_col = true_y.unsqueeze(0).repeat(b, 1)
    pos_mask = torch.eq(y_row, y_col).float().to(y_row.device)
    pos_mask = pos_mask * y_mask_map * (1 - torch.eye(b, device=rep.device))
    # print(f'pos_mask = \n{pos_mask}')

    # print("max element in sim_mat", torch.max(sim_mat))
    # print("temperature", temperature)
    # print("dt_m * temperature", dt_m * temperature)
    exp_a = (sim_mat / temperature).exp() * y_mask_map
    exp_b = (sim_mat / (dt_m * temperature)).exp() * y_mask_map
    # print("max element in exp_a", torch.max(exp_a))
    # assert(not torch.isinf(exp_a).any())
    # assert(not torch.isinf(exp_b).any())
    # print(f'pos_mask = \n{pos_mask}')
    pos_a = exp_a * pos_mask
    pos_b = exp_b * pos_mask
    # assert(not torch.isnan(pos_a).any())
    # assert(not torch.isnan(pos_b).any())
    # assert(torch.max(exp_a - pos_a) > 0)
    sum_a = exp_a.sum(dim=1)
    sum_b = exp_b.sum(dim=1)
    sum_pos_a = pos_a.sum(dim=1)
    sum_pos_b = pos_b.sum(dim=1)
    # assert(not torch.isnan(sum_b).any())
    # assert(not torch.isnan(sum_pos_b).any())
    fraction_over = (1 - sum_pos_b/(sum_b + 1e-9) + 1e-9)
    fraction_under = (1 - sum_pos_a/(sum_a + 1e-9) + 1e-9)
    # for i in range(fraction_over.shape[0]):
    #     if torch.isnan(fraction_over[i]):
    #         print(f"fraction_over[{i}] is nan")
    #         print(f"sum_pos_b[{i}]={sum_pos_b[i]}")
    #         print(f"sum_b[{i}]={sum_b[i]}")
    #         print(f"exp_b[{i}]={exp_b[i]}")
    #         print(f"pos_b[{i}]={pos_b[i]}")
    # assert(not torch.isnan(fraction_over).any())
    # assert(not torch.isnan(fraction_under).any())
    # print(f'fraction_over={fraction_over}, fraction_under={fraction_under}')
    coef = fraction_over / fraction_under
    coef = coef.detach()
    # print(f'coef={coef}')
    # assert(not torch.isnan(coef).any())

    nonzero = torch.nonzero(sum_pos_a).squeeze()
    sum_pos_a = sum_pos_a[nonzero]
    sum_a = sum_a[nonzero]
    coef = coef[nonzero]

    loss = -torch.log(sum_pos_a / (sum_a+1e-9)) * coef
    y_mask = y_mask[nonzero]
    loss = loss * y_mask
    loss = loss.sum() / y_mask.sum()
    return loss

    # intra-anchor hardness-awareness
