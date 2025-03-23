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
  - Arc consistency
    + AC-3
  - Backtracking search

## Learning  
  - Supervised learning
    + Given a data set of input-output pairs, learn a function to map inputs to outputs
  - Nearest-Neighbor Classification
  - Perceptron learning
  - Support Vector Machines
  - Regression
  - Loss functions
  - Overfitting
  - Regularization
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