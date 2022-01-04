class Node:

    def __init__(self, prior: float):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0 #노드의 평균값을 구해주기 위한 '합'의 값
        self.children = {}
        self.hidden_state = None #노드를 이동하면서 받게 되는 예상되는 보상값
        self.reward = 0

    def expanded(self) -> bool:
        return len(self.children) > 0
    
    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


    def expand_node():


    def 