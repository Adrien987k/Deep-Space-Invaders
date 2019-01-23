Deep-Space-Invaders
===================

Our project.
------------ 

We have a simple architecture : three convolutional layers + 2 parralel set of two fully connected layer, one computing V, the value of states, that is how much good it is to be in that state, one computing A(s,a), that represent how good it is to take action a in state s.

This modification of the basic architecture 3 cn + 2 fc is called Dueling DQN. (DQN stands for Deep Q-Network)

In terms of learning, we learn by looking at memories that we sample from a pool. That allows to not unlearn things that you learned in the past if the environnement changes (like going in a new level in some video games) or in the present case being hidden behind a shield or not.

The sampling isn't done uniformly, we use an optimization called PER : we prioritize experiences that have a huge loss, because there is much more to learn from them. The sampling is still done at random of course. But since that introduces a bias (we may learn too much from certain samples, which might lead to overfitting) so we dicrease the priority of experiences according to how much we did those.
