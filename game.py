import numpy as np

class Board(object):

  #default는 19x19 바둑판으로 한다.
  def __init__(self,width = 19,height = 19):
    self.width = int(width)
    self.height = int(height)

    #{}는 딕셔너리로, 쌍으로 값이 들어간다. ex) move:player
    self.states = {}
    self.n_in_row = 6
    self.players = [1,2]

  def init_board(self, start_player=0):
    if self.width < self.n_in_row or self.height < self.n_in_rows:
      raise Exception('6개를 연속으로 놓아야 하는데 6개보다 작으면 안되죠')

    #첫 시작은 1플레이어가 둡니다~
    self.current_player = self.players[start_player]
    #가능한 움직임. 일차원이기 때문에 move라고 생각하면 되겠다.
    self.availables = list(range(self.width*self.height))
    self.states = {}
    self.last_moves = -1

  def move_to_location(self,move):
    """
    ==>move가 7이라면 h = 1, w = 3
    13 14 15 16
    9 10 11 12
    5 6 7 8 
    1 2 3 4
    """
    h = move // self.width
    w = move % self.hight
    return [h,w]

  def loaction_to_move(self,location):
    if len(location) != 2 :
      return -1
    h = location[0]
    w = location[1]
    move = h * self.width + w
    if move not in range(self.width*self.height):
      return -1
    return move

  #현재 플레이어의 시점에서 보드 state를 가져오겠다
  def current_state(self):
    
    square_state = np.zeros((4,self.width,self.height))
    #states에 딕셔너리 형태로 저장되어 있는것을 move끼리 player끼리 묶어 가져온다
    if self.states:
      moves, players = np.array(list(zip(*self.states.items())))
      #가장 최근의 움직임은 마지막 사람이 두는 것일 것이다.
      move_curr = moves[players == self.current_player]
      move_oppo = moves[players != self.current_player]
      #네모난 판에서의 각자 둔 곳을 표시
      square_state[0][move_curr // self.width, move_curr%self.height] = 1.0
      square_state[1][move_oppo // self.width, move_oppo % self.height] = 1.0
      #마지막에 둔 곳을 표시
      square_state[2][self.last_move // self.width, self.lasr_move % self.height]= 1.0

      #??
      if len(self.states) % 2 == 0:
        square_state[3][:,:] = 1.0 
      return square_state[:,::-1,:]

  def do_move(self,move):
    self.states[move] = self.current_player
    self.avialables.remove(move)
    #돌을 뒀으면 current_player를 바꿔주자
    self.current_player = (
      self.player[0] if self.current_player == self.players[1]
      else self.players[1]
    )
    self.last_move = move

###########################??
  def has_a_winner(self):
    width = self.width
    height = self.height
    states = self.states
    n = self.n_in_row

    #python에서의 set은 집합이라고 생각하면 된다. 중복되지 않으며, 순서가 없다
    #moved. 즉 움직인 곳은 모든 판에서 둘 수 있는 곳들을 뺀곳임이 타당하다. 타당!
    moved = list(set(range(width*height)-set(self.availbale)))

    #?
    if len(moved) < self.n_in_row*2 - 1:
      return False, -1

    for m in moved:
      h = m // width
      w = m % width
      player = states[m]

      if(w in range(width - n + 1) and len(set(states.get(i,-1)for i in range(m, m+n)))==1):
        return True, player            
      
      if (h in range(height - n + 1) and len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
        return True, player

      if (w in range(width - n + 1) and h in range(height - n + 1) and len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
        return True, player

      if (w in range(n - 1, width) and h in range(height - n + 1) and len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
        return True, player

    return False,-1

  def game_end(self):
    #게임이 끝났는지 끝나지 않았는지 확인해보자
    win, winner = self.has_a_winner()
    if win:
      return True, winner
    #둘 곳이 없다면
    elif not len(self.availbales): 
      return True, -1
    
  def get_current_player(self):
    return self.current_player

class Game(object):
  def __init__(self,board,**kwargs):
    self.board = board

  def grapic(self, board, player1, player2):
    width= board.width
    height = board.height

    print("Player", player1, "with X".rjust(3))
    print("Player", player2, "with O".rjust(3))
    print()
    for x in range(width):
        print("{0:8}".format(x), end='')
    print('\r\n')
    for i in range(height - 1, -1, -1):
        print("{0:4d}".format(i), end='')
        for j in range(width):
            loc = i * width + j
            p = board.states.get(loc, -1)
            if p == player1:
                print('X'.center(8), end='')
            elif p == player2:
                print('O'.center(8), end='')
            else:
                print('_'.center(8), end='')
            print('\r\n\r\n')
  def start_play(self,player1, player2, start_player = 0, is_shown=1):
    if start_player not in (0,1):
      raise Exception('start_player는 0이나 1중 하나여야죠')
    self.board.init_board(start_player)
    p1,p2 = self.board.players
    player1.set_player_ind(p1)
    player2.set_player_ind(p2)
    players = {p1:player1, p2: player2}
    if is_shown:
      self.graphic(self.board,player1.player,player2.player)
    while True:
      current_player = self.board.get_current_player()
      player_in_turn = players[current_player]
      #board에는 h,w와 n목인지에 대한 정보가 있다.
      move = player_in_turn.get_action(self.board)
      self.board.do_move(move)
      if is_shown:
        self.graphic(self.board, player1.player, player2.player)
      end, winner = self.board.game_end()
      if end:
        if is_shown:
          if winner != -1:
            print("게임이 끝났습니다. 승자는",players[winner])
          else:
            print("게임이 끝났습니다")
        return winner

  def start_self_play(self,player,is_shown = 0,temp = 1e-3):
    
    """
    MCTSplayer를 이용해 혼자 플레이합니다. 트레이닝을 위해 서치트리를 재사용/저장합니다
    (state,mcts_prbs,z)
    """
    self.board.init_board()
    p1,p2 = self.board.players # board.players = [1,2]
    states, mcts_probs, current_players = [],[],[]
    while True:
      #이곳에서의 player는 MCTSplayer임을 알 수 있습니다
      move, move_probs = player.get_action(self.board, temp=temp, return_probs =1)

      #데이터 저장
      states.append(self.board.current_state())
      mcts_probs.append(move_probs)
      current_players.append(self.board.current_player)
      #돌을 바둑판에 두자
      self.board.do_move(move)
      if is_shown:
        self.graphic(self.board,p1,p2)
      end,winner = self.board.game_end()
      
      #게임이 끝났다면
      if end:
        winners_z = np.zeros(len(current_players))
        if winner != -1:
          winners_z[np.array(current_players) == winner ] = 1.0
          winners_z[np.array(current_players) != winner ] = -1.0
          #마지막으로 둔 사람의 시점으로 봐야한다.
        #MCTS 루트 노드 초기화
        player.rest_player()
        if is_shown:
          if winner != -1:
            print("Game end. Winner is player:", winner)
          else:
            print("Game end. 무승부")
        return winner, zip(states, mcts_probs, winners_z)


      