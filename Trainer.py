import os
import numpy as np
from random import shuffle
from game import Board,Game
import torch
import torch.optim as optim
from collections import defaultdict, deque
from model import PolicyValueNet
import mcts_aplhaZero import MCTSPlayer
from MCTS import MCTS

class TrainPipeline():
  def __init__(self,init_model=None):
    self.board_width = 19
    self.board_height = 19
    self.n_in_row = 6 #육목
    self. board = Board(width=self.board_width, height = self.board_height, n_in_row = self.n_in_row)
    self.game = Game(self.board)

    #파라미터를 학습시키자
    self.learn_rate = 2e-3  #최적의 learningRate을 찾아 볼 것
    self.lr_multiplier = 1.0 #학습속도 조절
    self.temp = 1.0 #temperature parameter
    self.n_playout = 400
    self.c_puct = 5
    self.buffer_size = 10000
    self.batch_size = 512 
    self.data_buffer = deque(maxlen=self.buffer_size)
    self.play_batch_size = 1
    self.epochs = 5
    self.kl_targ = 0.02
    self.check_freq = 50
    self.game_batch_num = 1500
    self.best_win_ratio = 0.0
    self.pure_mcts_playout_num = 1000

    #초기화 모델이 있다면 가져온다.
    if init_model:
      self.policy_value_net = PolicyValueNet(self.board_width,self.board_height,model_file=init_model)
    else:
      #초기화 모델이 없다면 새로운 policy-value net으로 트레이닝 시킨다.
      self.policy_value_net = PolicyValueNet(self.board_width,self.board_height)
      #policy_value_net을 적용한 mcts_player 생성
      self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,c_puct=self.c_puct,n_playout = self.n_playout,is_selfplay=1)


  def get_equi_data(self,play_data):
    """
    데이터를 넣어주면 데이터가 복사가 된다고?
    dataset을 반복시키거나 뒤집어서 dataset을 늘린다.
    play_data = [(바둑판의 상황[state],mcts돌렸을때 probs[mcts_probs],실제 승자[winner_z])가 배열로 들어있음]
    """
    extend_data = []
    for state, mcts_probs, winner in play_data:
      for i in [1,2,3,4]:

        #시계방향으로 돌려 데이터 수를 늘리자
        #i가 1이면 90도 2면 180도..
        equi_state = np.array([np.rot90(s,i) for s in state])
        equi_mcts_probs = np.rot90(np.flipud(mcts_prob.reshape(self.board_height,self.board_width)),i)  #상하 반전  
        extend.data.append((equi_state,np.flipud(equi_mcts_probs).flatten(),winner))

        #수평으로 뒤집자
        equi_state = np.array([np.fliplr(s) for s in equi_state])
        equi_mcts_probs = np.fliplr(equi_mcts_prob) #좌우 반전
        extend_data.append((equi_state,np.flipud(equi_mcts_prob).faltten(),winner))  
    
    return extend_data
  
  def collect_selfplay_data(self, n_games = 1):
    #트레이닝을 위한 솔로 플레이 데이터를 모으자 
    for i in range(n_games):
      #start_self_play로 혼자 게임 가능한 바둑판을 생성. start_self_play의 mctsplayer가 temp로 다음 action을 가져옴
      winner,play_data = self.game.start_self_play(self.mcts_player, temp=self.temp)
      play_data = list(play_data)[:]
      self.episode_len = len(play_data) #게임이 끝나면 1에피소드다.
      
      play_data = self.get_eqi_data(play_data)
      self.data_buffer.extend(play_data)

  def policy_update(self):
    #policy-value net을 업데이트 합시다.
    mini_batch = random.sample(self.data_buffer, self.batch_size) #데이터를 배치 사이즈만큼 랜덤으로 뽑는다
    state_batch = [data[0] for data in mini_batch]
    winner_batch = [data[2] for data in mini_batch] #state랑 winner랑 따로 묶어주고
    old_probs, old_v = self.policy_value_net.policy_value(state_batch)  #바둑판의 상황을 DNN에 넣었을 때 각 바둑판의 probs와 value를 가져온다
    for i in range(self.epochs):
      loss, entrophy = self.policy_value_net.train_step(state_batch,mcts_probs_batch,winner_batch,self.learn_rate*self.lr_multiplier)
      new_probs, new_v = self.policy_value_net.policy_value(state_batch)
      kl = np.mean(np.sum(old_probs * (
        np.log(old_probs + 1e-10)-np.log(new_probs + 1e-10)
      ),axis=1))
      if kl > self.kl_targ: * 4 #만약 D_KL divergence가 나쁘면 빠르게 멈추자
        break

      if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
        self.lr_multiplier /= 1.5
      elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
        self.lr_multiplier *= 1.5

      explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
      explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))
      print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        return loss, entropy
  
  def policy_evaluate(self, n_games=10):
        """
        학습된 policy를 pureMCTS 플레이어와 게임해 성적이 어떤지 확인합시다
        """
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout)
        pure_mcts_player = MCTS_Pure(c_puct=5,
                                     n_playout=self.pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(current_mcts_player,
                                          pure_mcts_player,
                                          start_player=i % 2,
                                          is_shown=0)
            win_cnt[winner] += 1
        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
                self.pure_mcts_playout_num,
                win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio

    def run(self):
        try:
            for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                print("batch i:{}, episode_len:{}".format(
                        i+1, self.episode_len))
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                # check the performance of the current model,
                # and save the model params
                if (i+1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i+1))
                    win_ratio = self.policy_evaluate()
                    self.policy_value_net.save_model('./current_policy.model')
                    if win_ratio > self.best_win_ratio:
                        print("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        # update the best_policy
                        self.policy_value_net.save_model('./best_policy.model')
                        if (self.best_win_ratio == 1.0 and
                                self.pure_mcts_playout_num < 5000):
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0
        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
    training_pipeline = TrainPipeline()
    training_pipeline.run()