## Chef's Hat Players Club
The Chef's Hat Player club provides ready-to-use agents for the Chef's Hat game.

## Instaltion

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
agent = DQL.DQL(name="DQL4", continueTraining=False, type="vsEveryone", initialEpsilon=1, verbose=True)  # training agent
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


### Karma Chameleon Club

Implementation of an Adversarial Inverse Reinforcement Learning Agent (AIRL), and instances of trained agents based on human data collected from Chef's Hat Online.

Types | Types | Types | Types | Types
------------ |------------ |------------ | ----------- | ----------- |
lilAbsol |lilDana |lilJBA |lilRamsey | lilYves | 
lilAle |lilDJ |lilLena |lilRaos |
lilAuar |lilDomi948 |lilLordelo |lilThecube |
lilBlio |lilEle |lilMars |lilThuran |
lilChu |lilFael |lilNathalia |lilTisantana |
 lilDa48 |lilGeo |lilNik |lilWinne |


Each agent can be instantiated using:

```python
agent = AIRL.AIRL(name="AIRL", continueTraining=False, type="lilDJ", initialEpsilon=1, verbose=True)  # training agent
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


## Citations

- Barros, P., Tanevska, A., Yalcin, O., & Sciutti, A. (2020). Incorporating Rivalry in Reinforcement Learning for a Competitive Game. arXiv preprint arXiv:2011.01337.
  
- Barros, P., Sciutti, A., Bloem, A. C., Hootsmans, I. M., Opheij, L. M., Toebosch, R. H., & Barakova, E. (2021, March). It's Food Fight! Designing the Chef's Hat Card Game for Affective-Aware HRI. In Companion of the 2021 ACM/IEEE International Conference on Human-Robot Interaction (pp. 524-528).

- Barros, P., Sciutti, A., Hootsmans, I. M., Opheij, L. M., Toebosch, R. H., & Barakova, E. (2020) The Chef's Hat Simulation Environment for Reinforcement-Learning-Based Agents. arXiv preprint arXiv:2003.05861.

- Barros, P., Tanevska, A., & Sciutti, A. (2020). Learning from Learners: Adapting Reinforcement Learning Agents to be Competitive in a Card Game. arXiv preprint arXiv:2004.04000.


## Contact

Pablo Barros - pablo.alvesdebarros@iit.it

- [http://pablobarros.net](http://pablobarros.net)
- [Twitter](https://twitter.com/PBarros_br)
- [Google Scholar](https://scholar.google.com/citations?user=LU9tpkMAAAAJ)