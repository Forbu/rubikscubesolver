# rubikscubesolver
Solviing rubik's cube with transformer world model

The idea is to use new architecture and RL algorithm around transformer and world model to solve rubik's cube.

Inspired from : 

- DO TRANSFORMER WORLD MODELS GIVE BETTER POLICY GRADIENTS ? (https://arxiv.org/pdf/2402.05290)  

- TRANSFORMER-BASED WORLD MODELS ARE HAPPY WITH 100K INTERACTIONS (https://arxiv.org/pdf/2303.07109)



#### Steps in order to train a proper world model

This is the overall training architecture :

![training](images/trainingsetup.png)

We can first train a world model with data gather from random experience (or policy experience).

Then we can train a policy model with the loss directly being the sum of reward.

#### Differents experiences

- Gumble softmax gives really unstable gradiant.

- Better have a model trained on policy probabilities than exact results (for world model)

- Should we be using a causal transformer for the policy model too ? Doesn't work great

- Forcing the WM to train only for non proba (High scale / temperature) degrade the performance



