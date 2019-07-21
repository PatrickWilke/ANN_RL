# ANN_RL
Project for testing different reinforcement learning algorithms in combination with artificial neural networks

This project is based on two recent seminars on machine learning and reinforcement learning. 
The goal is to combine these two concepts in simple interactive games which allow to get a direct impression of the 
learning success. The games, so far TicTacToe and Four in a Row, have GUIs which are implemented in kivy. 
They can be used to play against the previously trained AI. 
All trainings should be doable on a normal CPU within an hour such that everybody is able to perform them on their laptops.
The focus of this project is the approximation of state-action values using ANNs. For now it is restricted to dense NNs. 
Convoluted networks etc. could be used for more complex tasks but this is not too relevant right now. 
Here, I am interested in the effects of function approximation in combination with different learning algorithms in different
environments and how ANNs can be used to create a completely abstract framework.
As a start, I use (close to) on policy training. Later on I want to investigate the limits of off policy learning
in combination with function approximation by ANNs and bootstrapping. 
