# Disentangled representations and autonomous goal setting
Experiments carried out during the research stay in Innsbruck in order to continue working on the issue of disentangled representations and autonomous goal setting.

The proposal is the following:

1- Encoding world state representations into latent space and thus compressing the essential world properties depicted in the training image dataset.

2- After that, incorporating a Forward Model in between encoder and decoder, to predict next state in the series of temporally arranged images, to see if forward model is able to predict meaningful representations (even if they are not interpretable by visual inspection).

3- Once learned the new latent space and, having the Forward Model, be able to discover goals in the new latent space and learn Value Functions & Policies that allow the robot to reach them regardless of the initial position of these goals and the robot.

The figure shows a diagram of this process (steps 1 and 2).

![ExperimentalSetup](https://user-images.githubusercontent.com/25464222/142453068-026aee22-e3c7-4ca5-982d-f30576fa4840.PNG)


The idea behind this is to be able to carry out all the processes that we carry out in our architecture but starting from the original image, so that it is the robot itself that autonomously obtains the necessary representation. Since, in all our experiments, we established this representation manually in the form of distances, angles, etc. with the objects that were on the table.
