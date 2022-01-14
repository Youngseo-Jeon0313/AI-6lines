import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

def set_learning_rate(optimizer, lr):
  #주어주는 종류에 따라 lr를 세팅한다.
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr

class Net(nn.Module):
  def __init__(self, board_width, board_height):

    super(Net, self).__init__()

    self.board_width=board_width #19
    self.board_height=board_height #19

    #common layers
    self.conv1 = nn.Conv2d(4, 32, kernel_size = 3, padding=1) #in_channel, out_channel, kernel size, padding을 각각 할당
    #in_channel에 들어가는 것은 N, Q, u, P 이렇게 4개이다.

    self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, padding=1)
    self.conv3 = nn.Conv2d(64, 128, kenel_size = 3, padding=1)

    #action policy layers 
    #늘린 것을 다시 줄이는 작업 
    self.act_conv1 = nn.Conv2d(128, 4, kernel_size =1)
    self.act_fc1 = nn.Linear(4*board_width*board_height, board_width*board_height) #linear함수를 이용해서 한 grid당 값은 하나로 짠 하고 나오도록 바꾼다.

    #state value layers
    self.val_conv1 = nn.Conv2d(128, 2, kernel_size1=1)
    self.val_fc1 = nn.Linear(2*board_width*board_height, 64)
    self.val_fc2 = nn.Linear(64, 1) #value에 해당하는 layer는 2개에서 ... 1개의 값으로 나오도록 한다.
    
  def forward(self, state_input):
    """
   foward() 함수는 모델이 학습데이터를 입력받아서 forward 연산을 진행시키는 함수.
   이 forward() 함수는 model 객체를 데이터와 함께 호출하면 자동으로 실행됨
    """
    x=F.relu(self.conv1(state_input)) #relu함수로 걸러준다.
    x=F.relu(self.conv2(x)) #각각 계속 conv1, 2, 3을 차례대로 실행
    x=F.relu(self.conv3(x))

    #그 다음 층은 action_policy 관련
    #action policy layers 
    x_act = F.relu(self.act_conv1(x)) 
    x_act = x_act.view(-1, 4*self.board_width*self.board_height) #차원을 하나 줄인다.
    x_act = F.log_softmax(self.act_fc1(x_act)) #softmax함수에 넣어줌.

    #state value layers
    x_val = F.relu(self.val_conv1(x))
    x_val = x_val.view(-1, 2*self.board_width*self.board_height)
    x_val = F.relu(self.val_fc1(x_val))
    x_val = F.tanh(self.val_fc2(x_val))
    
    return x_act, x_val
  
class PolicyValueNet():
  def __init__(self, board_width, board_height, model_file = None, use_gpu=False):
    self.use_gpu = use_gpu
    self.board_width = board_width
    self.board_height = board_height
    self.l2_const=1e-4 #l2 정규화 :overfitting을 막기 위한 c세타 제곱이라고 생각

    if self.use_gpu:
      self.policy_value_net = Net(board_width, board_height).cuda() 
      #Net에 해당하는 요소들 중 width랑 height를 이용할 예정
    else:
      self.policy_value_net = Net(board_width, board_height)
    self.optimizer = optim.Adam(self.policy_value_net.parameters(), weight_decay=self.l2_const)

    if model_file : #모델 파일이 있다면
      net_params = torch.load(model_file)
      self.policy_value_net.load_state_dict(net_params)

  def policy_value(self, state_batch):
    '''
    input: state 배치
    output: action_probs 랑 state values 배치
    '''
    if self.use_gpu:
    #맞으면 gpu에서 가져오고
      state_batch = Variable(torch.FloatTensor(state_batch).cuda())
      #Variable은 Tensor랑 같은 거라고 생각하면 됨. 연산 그래프에서 Node로 표현됨
      log_act_probs, value= self.policy_value_net(state_batch)
      #Net에 넣어 나온 각각의 log_action_probs와 value가 각 grid에 나오게 된다.
      act_probs = np.exp(log_act_probs.data.cpu().numpy())
      #.data를 이용해서 값을 가져와 cpu에 넣음
      return act_probs, value.data.cpu().numpy()
    else:
    #아니면 그냥 값을 가져온다.
      state_batch = Variable(torch.FloatTensor(state_batch)) #또 노드에 넣어줌
      log_act_probs, value = self.policy_value_net(state_batch)
      act_probs = np.exp(log_act_probs.data.numpy())
      return act_probs, value.data.numpy()

  def policy_value_fn(self, board):
    '''
    input: board
    output : board당 action, probability, state의 점수 의 list
    '''
    legal_positions = board.availables #가능한 board판의 위치를 각각 가져옴
    current_state = np.ascontiguousarray(board.current_state().reshape(-1, 4, self.board_width, self.board_height)) #차원 하나를 줄여준다. ascontiguousarray = 메모리 배열 연속적으로 반환시킴 [1,2,3] => [1 2 3] 이런 형식으로 
    
    if self.use_gpu :
      log_act_probs, value = self.policy_value_net(Variable(torch.from_numpy(current_state)).cuda().float())
    else:
      log_act_probs, value = self.policy_value_net(Variable(torch.from_numpy(current_state)).float())
      act_probs = np.exp(log_act_probs.data.numpy().flatten())
      #가져와서 flatten함수로 차원 눌러줌
    act_probs = zip(legal_positions, act_probs[legal_positions])
    #iterable를 인자로 받고, 각 객체가 담고 있는 원소를 터플의 형태로 차례로 접근할 수 있는 반복자(iterator)를 반환
    value = value.data[0][0]
    return act_probs, value

  def train_step(self, state_batch, mcts_probs, winner_batch, lr):
    if self.use_gpu:
      state_batch=Variable(torch.FloatTensor(state_batch).cuda())
      mcts_probs = Variable(torch.FloatTensor(mcts_probs).cuda())
      winner_batch = Variable(torch.FloatTensor(winner_batch).cuda())
    else:
      state_batch = Variable(torch.FloatTensor(state_batch))
      mcts_probs = Variable(torch.FloatTensor(mcts_probs))
      winner_batch = Variable(torch.FloatTensor(winner_batch))
    
    
    self.optimizer.zero_grad()
    #zero_grad로 optimizer 설정
    set_learning_rate(self.optimizer, lr)

    #forward로 model 틀 가져오기
    log_act_probs, value = self.policy_value_net(state_batch)

    # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
    value_loss = F.mse_loss(value.view(-1), winner_batch)
    policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, 1))
    loss = value_loss + policy_loss

    #backward and optimize
    loss.backward()
    self.optimizer.step()
    entropy = -torch.mean(torch.sum(torch.exp(log_act_probs)*log_act_probs, 1))
    
    return loss.data[0], entropy.data[0]

def get_policy_param(self):
  net_params = self.policy_value_net.state_dict()
  return net_params

def save_model(self, model_file)
  net_params = self.get_policy_params()
  torch.ave(net_params, model_file)
  #model저장시키는 함수

#??loss표현 방식이 다름 ㅠ
