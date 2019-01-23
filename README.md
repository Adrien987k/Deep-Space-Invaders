Deep-Space-Invaders
===================

Our project.
------------ 

We have a simple architecture : three convolutional layers + 2 parralel set of two fully connected layer, one computing V, the value of states, that is how much good it is to be in that state, one computing A(s,a), that represent how good it is to take action a in state s.

We take as input a modified version of the screen (we scrap some uninteresting part, we take it in black and white to reduce dimensions etc...)

This modification of the basic architecture 3 cn + 2 fc is called Dueling DQN. (DQN stands for Deep Q-Network)

In terms of learning, we learn by looking at memories that we sample from a pool. That allows to not unlearn things that you learned in the past if the environnement changes (like going in a new level in some video games) or in the present case being hidden behind a shield or not.

The sampling isn't done uniformly, we use an optimization called PER : we prioritize experiences that have a huge loss, because there is much more to learn from them. The sampling is still done at random of course. But since that introduces a bias (we may learn too much from certain samples, which might lead to overfitting) so we decrease the priority of experiences according to how much we did those.

One problem with Deep Q-learning is that the target we are trying to reach is moving at the same time we progress since it's supposedly computed with the network, so we use a trick called Q-fixed target that adds a network that computes the target but is updated less frequently than the 'main' network. We also use this second network to decouple the choice of action and evaluation of q-valued (dueling dqn).

sources :

https://medium.freecodecamp.org/an-introduction-to-deep-q-learning-lets-play-doom-54d02d8017d8
https://medium.freecodecamp.org/improvements-in-deep-q-learning-dueling-double-dqn-prioritized-experience-replay-and-fixed-58b130cc5682
The paper in ressources.
