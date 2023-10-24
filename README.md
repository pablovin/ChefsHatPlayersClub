## Chef's Hat Players Club
The Chef's Hat Player club provides ready-to-use agents for the Chef's Hat game.

## Instalation

The instalation can be done via pip:

```python
pip install ChefsHatPlayersClub
```

##Available Agents

Agents are divided into types, currently these are the types of available agents. You can check th examples folder for more examples!

### Classic Agents

The classic agents are the ones available in our initial investigations with the Chef's Hat:
- Barros, P., Tanevska, A., & Sciutti, A. (2020). Learning from Learners: Adapting Reinforcement Learning Agents to be Competitive in a Card Game. arXiv preprint arXiv:2004.04000.

Each type of the agent refers on how the agent was trained. Refer to the above paper for more information.

Agent | Types |
------------ | -------------
DQL | "Scratch", "vsRandom", "vsSelf", "vsEveryone" 
PPO | "Scratch", "vsRandom", "vsSelf", "vsEveryone"

Each agent can be instantiated using:

```python
agent = AgentDQL(name="DQL4", continueTraining=False, type="vsEveryone", initialEpsilon=1, verbose=True, logDirectory="")  # training agent
```
where the initialization parameters are:

Parameter | Description |
------------ | -------------
name | name of the agent, must be unique in a game
continueTraining | if the agent will learn during the game
type | type of the agent, see table above
initialEpsilon| when learning, initial exploration value
saveFolder| folder that the agent will be saved in
verbose | verbose when learning
logDirectory | The directory where the agent log will be located


### Karma Chameleon Club

Implementation of an Adversarial Inverse Reinforcement Learning Agent (AIRL), and instances of trained agents based on human data collected from Chef's Hat Online.


| Type | Type | Type     | Type     | Type |
|-----------|----------|------------|------------|-----------|
| lil_abcd_ | lilAbsol | lilAle     | lilAna     | lilArkady |
| lilAuar   | lilBlio1 | lilBlio2   | lilChu     | lilDa48   |
| lilDana   | lilDJ    | lilDomi948 | lilEle     | lilFael   |
| lilGeo    | lilIlzy  | lilJba     | lilLeandro | lilLena   |
| lilLordelo    | lilMars  | lilNathalia     | lilNik | lilNilay   |
| lilRamsey    | lilRaos  | lilThecube     | lilThuran | lilTisantana   |
| lilToran    | lilWinne  | lilYves     | lilYves2 |    |

Each agent can be instantiated using:

```python
agent = AgentAIRL(name="AIRL", continueTraining=False, type="lilDJ", initialEpsilon=1, verbose=True, logDirectory="")  # training agent

```
where the initialization parameters are:

Parameter | Description |
------------ | -------------
name | name of the agent, must be unique in a game
continueTraining | if the agent will learn during the game. Must have a demonstration if continue learning is True.
demonstrations | must be a npy with a list of ([state, action, possibleActions])
type | type of the agent, see table above
initialEpsilon| when learning, initial exploration value
saveFolder| folder that the agent will be saved in
verbose | verbose when learning
logDirectory | The directory where the agent log will be located

### Chef`s Hat Cup 1

Implementations of the agents that won the Chef`s Hat Cup 1. For more information please visit: https://github.com/yvesmendes/ChefsHatPlayersClubAgents/

Agent | Description
------------  | ------------ 
AIACIMP |  use a custom loss function based on the Hub Loss and a custom reward
AINSA | Use a custom reward
ALLIN | Custom reward
AMYG4 | Only winning reward

where the initialization parameters are:

Parameter | Description |
------------ | -------------
name | name of the agent, must be unique in a game
continueTraining | if the agent will learn during the game. Must have a demonstration if continue learning is True.
demonstrations | must be a npy with a list of ([state, action, possibleActions])
initialEpsilon| when learning, initial exploration value
saveFolder| folder that the agent will be saved in
verbose | verbose when learning
logDirectory | The directory where the agent log will be located

```python
agent = AIACIMP(name="01", continueTraining=False, initialEpsilon=0.2, loadNetwork="", saveFolder="", verbose=True, logDirectory=logDirectory)
agent = AINSA(name="01", continueTraining=False, initialEpsilon=0.2, loadNetwork="", saveFolder="", verbose=True, logDirectory=logDirectory)
agent = ALLIN(name="01", continueTraining=False, initialEpsilon=0.2, loadNetwork="", saveFolder="", verbose=True, logDirectory=logDirectory)
agent = AMYG4(name="01", continueTraining=False, initialEpsilon=0.2, loadNetwork="", saveFolder="", verbose=True, logDirectory=logDirectory)
```


## Citations

- Barros, P., Yalçın, Ö. N., Tanevska, A., & Sciutti, A. (2023). Incorporating rivalry in reinforcement learning for a competitive game. Neural Computing and Applications, 35(23), 16739-16752.

- Barros, P., & Sciutti, A. (2022). All by Myself: Learning individualized competitive behavior with a contrastive reinforcement learning optimization. Neural Networks, 150, 364-376.

- Barros, P., Yalçın, Ö. N., Tanevska, A., & Sciutti, A. (2022). Incorporating Rivalry in reinforcement learning for a competitive game. Neural Computing and Applications, 1-14.

- Barros, P., Tanevska, A., & Sciutti, A. (2021, January). Learning from learners: Adapting reinforcement learning agents to be competitive in a card game. In 2020 25th International Conference on Pattern Recognition (ICPR) (pp. 2716-2723). IEEE.

- Barros, P., Sciutti, A., Bloem, A. C., Hootsmans, I. M., Opheij, L. M., Toebosch, R. H., & Barakova, E. (2021, March). It's Food Fight! Designing the Chef's Hat Card Game for Affective-Aware HRI. In Companion of the 2021 ACM/IEEE International Conference on Human-Robot Interaction (pp. 524-528).

- Barros, P., Tanevska, A., Cruz, F., & Sciutti, A. (2020, October). Moody Learners-Explaining Competitive Behaviour of Reinforcement Learning Agents. In 2020 Joint IEEE 10th International Conference on Development and Learning and Epigenetic Robotics (ICDL-EpiRob) (pp. 1-8). IEEE.

- Barros, P., Sciutti, A., Bloem, A. C., Hootsmans, I. M., Opheij, L. M., Toebosch, R. H., & Barakova, E. (2021, March). It's food fight! Designing the chef's hat card game for affective-aware HRI. In Companion of the 2021 ACM/IEEE International Conference on Human-Robot Interaction (pp. 524-528).


## Contact

Pablo Barros - pablovin@gmail.com

Alexandre Rodolfo - armp@ecomp.poli.br

- [Twitter](https://twitter.com/PBarros_br)
- [Google Scholar](https://scholar.google.com/citations?user=LU9tpkMAAAAJ)
