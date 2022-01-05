'''
game.reward - 게임의 각 턴에 받는 실제 보상 목록
game.history - 게임의 각 턴에 취한 행동 목록
game.child_visit - 게임의 각 턴에서 루트 노드의 행동 확률 분포 목록
game.root_values - 게임의 각 턴에서 루트 노드의 값 목록

'''

class Game:
    
    def __init__(self, game):
        self.reward= game.reward
        self.history = game.history
        self.child_visit = game.child_visit
        self.root_values = game.root_values