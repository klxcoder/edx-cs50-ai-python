# CS50's Introduction to Artificial Intelligence with Python  
[Course Link](https://learning.edx.org/course/course-v1:HarvardX+CS50AI+1T2020/home)  

## Contents  
- [x] [**Introduction**](#introduction)  
- [ ] [**Search**](#search)  
- [ ] [**Knowledge**](#knowledge)  
- [ ] [**Uncertainty**](#uncertainty)  
- [ ] [**Optimization**](#optimization)  
- [ ] [**Learning**](#learning)  
- [ ] [**Neural Networks**](#neural-networks)  
- [ ] [**Language**](#language)  

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
    Use in DFS
    Expand the deepest node in the frontier
  - Queue
    Fist-in first-out data type
    Use in BFS
    Expand the shallowest node in the frontier
  - Heuristic function
  - Greedy Best-Frist Search
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
      * minimax (1h15m09s)

## Knowledge  
(TBD)  

## Uncertainty  
(TBD)  

## Optimization  
(TBD)  

## Learning  
(TBD)  

## Neural Networks  
(TBD)  

## Language  
(TBD)  
