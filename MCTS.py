import torch
import math
import numpy as np

def ucb_score(parent, child):
    #가장 큰 값의 ucb를 가지는 것을 택한다.
    prior_score = child.prior *math.sqrt(parent.visit_count) /(child.visit_count +1)
    #맨 처음 score값은 일단 random으로 뽑을 수밖에 없는데, 그래도 이런 수식을 사용한다.
    if child.visit_count > 0: #만약 child를 방문한 흔적이 있다면
        value_score = -child.value() #child node를 바탕으로 value_score을 계산한다. child의 value는 -로 가져와줌..
    else:
        value_score = 0
    
    return value_score + prior_score #child의 value score과 parent/child의 N값을 이용해 최종 score을 구한다.

class Node:
    def __init__(self, prior, to_play):
        self.visit_count = 0
        self.to_play = to_play #player
        self.prior = prior  #prior_probability
        self.value_sum = 0  
        self.children = {}  #자식 노드 -dictionary 값으로 할당
        self.state = None
        
    def expanded(self): #자식 노드가 있는지를 확인하는 함수
        return len(self.children) > 0 
    
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count #sigma V / N
    
    def select_action(self, temperature):
        visit_counts = np.array([child.visit_count for child in self.children.values()])
        #values함수는 그 배열에 있는 값들을 일단 다 꺼내오는 것. 즉 자식 노드들 각각에 있는 N을 꺼내서 배열로 만든다.
        actions = [action for action in self.children.keys()] 
        #keys함수는 dictionary 형태로 되어 있는 배열 안에서 action을 각각 가져오는 것 - 확률값처럼 가져온다고 예측하였다.
        #Temperature parameter : 낮은 확률의 후보에 대한 민감도를 높임.
        if temperature == 0: 
            action = actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = np.random.choice(actions)
        else:
            visit_count_distribution = visit_counts ** (1/temperature)
            visit_count_distribution = visit_count_distribution / sum(visit_count_distribution)
            action = np.random.choice(actions, p=visit_count_distribution)
            #정규분포 형식처럼 N에 대한 정보를 각각 sum으로 나누어 나타내준다.
            #결국 여기에서 action을 아무리 random하게 뽑는다고 하더라도 더 확률값이 높은 action을 선택할 확률이 높아짐
        return action
    
    def select_child(self):
        #가장 높은 UCB 점수를 가진 자식을 뽑는 코드
        best_score = -np.inf #일단 -무한대로 best_score 초기화(off-policy 의 특징)
        best_action = -1
        best_child = None
        
        for action, child in self.children.items():
            #items는 파이썬에서 딕셔너리의 key/value값을 모두 가져온다.
            score = ucb_score(self, child) #이렇게 ucb를 이용해 가져온 score가
            if score > best_score:  #만약 지금까지의 최적 score보다 크다면 갱신
                best_score = score
                best_action = action
                best_child = child
        return best_action, best_child
    
    def expand(self, state, to_play, action_probs):
        # 노드를 확장해줄 때 코드.--DNN에서 학습한 결과 바탕으로
        self.to_play = to_play
        self.state = state
        for a, prob in enumerate(action_probs): #a는 children의 번째수, prob은 children으로 갈 확률
            if prob != 0:
                self.children[a] = Node(prior=prob, to_play = self.to_play *-1) #children 또한 하나의 노드로 생각해준다. children으로 하나 갈 때 게임 판을 상대에게 넘겨줌
                
    def __repr__(self):
        prior = "{0:.2f}".format(self.prior)
        return "{} Prior: {} Count: {} Value: {}".format(self.state.__str__(), prior, self.visit_count, self.value())

    
class MCTS:
    
    def __init__(self, game, model, args):
        self.game = game
        self.model = model #DNN model에서 가져옴
        self.args = args #인자
        
    def run(self, model, state, to_play):
        
        root = Node(0, to_play) #처음 root노드는 0(번째?)으로 지정되며 to_play하는 사람은 나..
        
        #root를 확장한다.
        action_probs, value, model.predict(state) #DNN에서 가져온다.
        valid_moves = self.game.get_valid_moves(state) #가능한 움직임들을 파악한다. 0-안돼 1-돼
        action_probs = action_probs * valid_moves # 가능한 움직임들에 action할 때의 확률들을 곱해준다.
        action_probs /= np.sum(action_probs) #확률값으로 바꾸어준다.
        root.expand(state, to_play, action_probs)
        
        for _ in range(self.args['num_simulations']):
            node = root
            search_path = [node]
            
            #SELECT
            while node.expanded(): #봤더니 노드가 확장되어 있어.
                action, node = node.select_child() #ucb가 가장 높은 child 뽑기
                search_path.append(node) #search_path에 이 node를 또 넣는다.
            
            parent = search_path[-2] #이렇게 root,child기준으로 간다면 parent는 바로 전전 꺼
            state = parent.state #그러면 state는 parent의 state로 넣어준다.
            
            #s'에 관한..!
            next_state, _ = self.game.get_next_state(state, player = 1, action=action)
            #게임 도중 이제 상대가 둔 이후의 상황이 s'이 된다.
            next_state = self.game.get_canonical_board(next_state, player=-1)
            #그러면 이제 판때기에서 상황 가져오기
            
            value = self.game.get_reward_for_player(next_state, player = 1)
            #value는 사실상 reward를 가져오느냐 마느냐인데 -1, 0, 1값으로 나타내지는 거임
            if value is None: #근데 아직 게임이 안 끝났다고 한다면
                #노드를 더 확장해야 한다.
                action_probs, value = model.predict(next_state)
                valid_moves = self.game.get_valid_moves(next_state)
                action_probs = action_probs * valid_moves
                action_probs /= np.sum(action_probs)
                node.expand(next_state, parent.to_play * -1, action_probs)
            
            self.backup(search_path, value, parent.to_play * -1)
            #만약 게임이 끝났다면(leaf node) DNN에서 가져오고, 아니라면 predict값을 가져옴
            
        return root
    
    def backup(self, search_path, value, to_play):
        for node in reversed(search_path): #역추적을 해나가면서
            node.value_sum += value if node.to_play == to_play else -value
            #지금이 '내'가 두는 거면 value를 value_sum에 더하고 아니면 -value를 더한다. 
            #역전 과정에서 남의 수 또한 판단하므로.!
            node.visit_count += 1
            #node의 visit 빈도수를 하나씩 높여준다.