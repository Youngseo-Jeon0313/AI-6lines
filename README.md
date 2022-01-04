# AI-육목

# 구현 계획

2022.01.04
alpha zero 논문 참고. 기계가 선생처럼 자신이 플레이하는 게임의 룰도 따로 학습하고 자신의 policy network도 꾸준히 update해 나가는 기계학습을 시킬 예정이다.
코드를 분류해보자면 일단,

//mcts.py
mcts tree 구조를 만든다.
tree를 방문할 때 expand, select, backpropagation, evaluation을 기준으로 코드를 나누어 작성한다.
너무 어려운 ..코드는 아직 초보단계인 나에게 너무 어려울 듯 하여 가장 이해가 잘되는 
http://joshvarty.github.io/AlphaZero/ 이 2목(??) 코드를 완전히 이해한 후 작성하려 한다.
아무래도 alphazero의 3가지 핵심 부분은
1. value network
2. policy network
3. monte carlo tree search 
이며, 이 때 3에다가 1,2를 어떻게 조화롭게 넣느냐에 초점을 맞추어 코드를 짤 생각이다.



