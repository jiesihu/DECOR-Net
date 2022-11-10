from torch import nn
from torch.nn import CrossEntropyLoss,Softmax
class decorrelation_loss(nn.Module):
  def __init__(self):
    super(Channel_loss, self).__init__()
    self.softmax = Softmax(dim=-1)
    self.loss = nn.CrossEntropyLoss()

  def forward(self, x):
#     x is the feature map with dimension [batchsize, channel, height, width]
    m_batchsize, C, height, width = x.size()
    proj_query = x.view(m_batchsize, C, -1)
    proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
    energy = torch.bmm(proj_query, proj_key)/height/width
    energy_norm = torch.div(energy, torch.max(energy, -1, keepdim=True)[0].expand_as(energy))
    label_made = torch.arange(0,C,1).expand(m_batchsize,-1).to(x.device)
    output = self.loss(energy_norm, label_made)
    return output
