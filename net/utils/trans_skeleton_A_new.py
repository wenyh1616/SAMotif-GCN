# import torch
# import torch.nn as nn
import jittor
import jittor.nn as nn
import matplotlib.pyplot as plt

class TransSkeleton(nn.Module):

    def __init__(self):
        super().__init__()

    def execute(self, x, A):

        N, C, T, V = x.size()
        x = x.permute(0, 2, 3, 1)
        x = x.view(N, T, V, C).mean(1)
        x_t = x.permute(0, 2, 1)

        vec_prod = jittor.matmul(x, x_t)
        sum_sqa = jittor.sum(x*x, dim=2)
        sum_sqa2 = jittor.sum(x*x, dim=2)
        sum_sqa_ex = sum_sqa.repeat(1, 1, vec_prod.shape[2]).view(-1, V, V)
        sum_sqb_ex = sum_sqa2.repeat(1, 1, vec_prod.shape[1]).view(-1, V, V) #.transpose(1,2)
        sum_sqb_ex = sum_sqb_ex.permute(0, 2, 1) #.contiguous()
        sq_ed = sum_sqb_ex + sum_sqa_ex - 2 * vec_prod
        zero_vec = jittor.zeros_like(sq_ed)
        ed = jittor.where(sq_ed > 0, sq_ed, zero_vec)

        ed = ed.view(N, V * V)
        max_ed = jittor.max(ed, 1)
        max_ed = max_ed.view(N, -1)
        ed = max_ed - ed
        ed = ed.view(N, V, V)
        #test = ed[0].cpu().detach().numpy()
        A_re = jittor.sum(A[:3], dim=0)
        zero_vec = jittor.zeros_like(ed)
        A_re = jittor.expand(A_re,(N,*A_re.size())) #待定
        ed = jittor.where(A_re > 0, zero_vec, ed)  

        x_diag = jittor.sum(ed, 1)
        x_diag = x_diag.repeat(1, V).view(-1, V, V)
        ed = jittor.divide(ed, x_diag)

        ed = ed.view(N, V, V)
        #test2 = ed[0].cpu().detach().numpy()
        # fig = plt.figure()
        # #
        # # cmap =  plt.cm.gray
        # cmap = plt.cm.cool
        # ax = fig.add_subplot(111)
        # im = ax.imshow(ed[0].cpu().detach().numpy(), cmap=cmap, vmin=0, vmax=0.2)
        # plt.colorbar(im, shrink=0.5, ticks=[0,0.1,0.2])
        # plt.show()
        return ed
