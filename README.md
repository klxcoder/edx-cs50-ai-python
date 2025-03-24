# CS50's Introduction to Artificial Intelligence with Python  
[Course Link](https://learning.edx.org/course/course-v1:HarvardX+CS50AI+1T2020/home)  

## Contents  
- [x] [**Introduction**](#introduction) [video](https://edx-video.net/3a64c0ea-1f2c-4121-9192-d91d5772830f-mp4_720p.mp4) 
- [x] [**Search**](#search) [video](https://youtu.be/D5aJNFWsWew)
- [ ] [**Knowledge**](#knowledge) [video](https://youtu.be/HWQLez87vqM)
- [ ] [**Uncertainty**](#uncertainty) [video](https://youtu.be/D8RRq3TbtHU)
- [x] [**Optimization**](#optimization) [video](https://youtu.be/qK46ET1xk2A)
- [ ] [**Learning**](#learning) [video](https://youtu.be/-g0iJjnO2_w)
- [ ] [**Neural Networks**](#neural-networks) [video](https://youtu.be/J1QD9hLDEDY)
- [ ] [**Language**](#language) [video](https://youtu.be/QAZc9xsQNjQ)

---

## Introduction
  - Short video introduce topics we will learn in the course

## Search
  - Long video: 1h49m29s
  - Introduce examples of realworld search algorithms such as Google Map
  - Introduce terms such as: Agent, State, Action
  - initial state
  - actions
  - Transition model: new_state = result(state, action)
  - state space: The set of all states reachable from the initial state by any sequence of actions
  - goal test: way to determine whether a given state is a goal state
  - path cost: numerical cost associated with a given path
  - path cost function
  - optimal solution: a solution that has the lowest path cost among all solutions
  - node: a data structure that keeps track of:
    + a state
    + a parent (node that generated this node)
    + an action (action applied to parent to get node)
    + a path cost (from initial state to node)
  - Approach
    + Start with a `frontier` that contains the inital state
    + Repeat:
      * If the frontier is empty, there is no solution
      * Remove the node from the frontier
      * If node contains goal state, return the solution
      * Expand node, add resulting nodes to the frontier
  - Revised Approach
    + Start with a `frontier` that contains the inital state
    + Start with with an empty explored set
    + Repeat:
      * If the frontier is empty, there is no solution
      * Remove the node from the frontier (stack or queue)
      * If node contains goal state, return the solution
      * Add the node to the expored set
      * Expand node, add resulting nodes to the frontier if they aren't already in the frontier or the explored set
  - Stack
    Last-in first-out data type
    Use in DFS (Depth First Search)
    Expand the deepest node in the frontier
  - Queue
    Fist-in first-out data type
    Use in BFS (Breadth First Search)
    Expand the shallowest node in the frontier
  - Uniform search
    + Search strategy that uses no problem-specific knowledge
  - Heuristic function
    + Manhattan distance
  - Greedy Best-Frist Search
    + Search algorithm that expands the node that is closet to the goal, as estimated by a heuristic function h(n)
  - A* search:
    + Search algorithm that expands node with lowest value of g(n) + h(n)
    + g(n) = cost to reach node
    + h(n) = estimated cost to goal
  - Adversarial search
    + Example games:
      * tic-tac-toe
      * chess
      * go
    + Search algorithm to use
      * minimax
      * alpha-beta pruning
      * Depth-limited minimax
      * Evaluation function

## Knowledge  
(TBD)  

## Uncertainty  
(TBD)  

## Optimization  
  - Local search
    + Search algorithms that maintain a single node and searches by moving to a neighboring node
  - Global maximum
    + objective function
  - Global minimum
    + cost function
  - Hill climbing
    + Local maxima
    + Local minima
    + Flat local maximum
    + Shoulder
    + Hill climbing variants
      * steepest-ascent: choose the highest-valued neighbor
      * stochastic: choose randomly from higher-valued neighbors
      * first-choice: choose the first higher-valued neighbor
      * random-restart: conduct hill climbing multiple times
      * local beam search: chooses the k highest-valued neighbors
  - Simulated annealing
    + Early on, higher "temperature": more likely to accept neighbors that are worse than current state
    + Later on, lower "temperature": less likely to accept neighbors that are worse than current state
  - Traveling salesman problem
  - Linear programming algorithms
    + Simplex
    + Interior-Point
  - Constraint satisfaction problem (CSP)
    + variables
    + domains
    + constraints:
      * hard constraints/soft constraints
      * unary constraint/binary constraint
  - Node consistency
  - Arc consistency (AC-3)
  - Backtracking search (brute force with constraint pruning)

## Learning  
  - Supervised learning
    + Given a data set of input-output pairs, learn a function to map inputs to outputs
    + Classification:
      * Supervised learning task of learning a function mapping an input point to a discrete category
  - Nearest-Neighbor Classification
    + algorithm that, given an input, chooses the class of the nearest data point to that input
  - k-nearest-neighbor classification
    + algorithm that, given an input, chooses the most common class out of the k nearest data points to that input
  - Perceptron learning
    + Perceptron learning rule
    + hard threshold/soft threshold
  - Support Vector Machines
    + maximum margin separator
      * boundary that maximizes the distance between any of the dat points
    + find the optimal `hyperplane` that best separates data points into different classes
  - Regression
    + Supervised learning task of learning a function mapping an input point to a continuos value
  - Loss functions
    + a function that expresses how poorly our hypothesis performs
    + Common loss functions
      * 0-1 loss function
      * L1 loss function: L(actual, predicted) = |actual - predicted|
      * L2 loss function: L(actual, predicted) = (actual - predicted)^2
  - Overfitting
    + a model that fits too closely to a particular data set and therefore may fail to generalize to future data
  - Regularization
    + penalizing hypotheses that are more complex to favor simpler, more general hypotheses
    + cost(h) = loss(h) + λ * complexity(h)
    + holdout cross-validation:
      * splitting data into a training set and a test set, such that learning happens on the training set and is evaluated on the test set
      <!-- Will continue from https://youtu.be/-g0iJjnO2_w?t=3415 -->
  - scikit-learn
  - Reinforcement Learning
  - Markov Decision Processes
  - Q-learning
  - Unsupervised Learning
  - k-means Clustering

## Neural Networks  
(TBD)  

## Language  
(TBD)  

## Files
  - https://cdn.cs50.net/