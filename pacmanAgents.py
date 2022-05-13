from pacman import Directions
from game import Agent
import random
import game
import util
import numpy as np


#reflex agent chooses action at each choice point by examing the alteranatives via a state evaluation function
class ReflexAgent(Agent):

    def getAction(self, gameState):
        #collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        #chooses one of the best actions available
        scores = [evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        #pick randomly among the best actions
        chosenIndex = random.choice(bestIndices)

        return legalMoves[chosenIndex]


#class provides common elements to all the multiagent searchers (minimax, and alphaBeta agents)
class MultiAgentSearchAgent(Agent):

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth='2'):
        self.index = 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


def getLegalActionsNoStop(index, gameState):
    possibleActions = gameState.getLegalActions(index)
    if Directions.STOP in possibleActions:
        possibleActions.remove(Directions.STOP)
    return possibleActions


class MinimaxAgent(MultiAgentSearchAgent):

    def minimax(self, agent, depth, gameState):
        if gameState.isLose() or gameState.isWin() or depth == self.depth:
            return self.evaluationFunction(gameState)
        if agent == 0: #maximize for pacman
            return max(self.minimax(1, depth, gameState.generateSuccessor(agent, action)) for action in
                       getLegalActionsNoStop(0, gameState))
        else: #minimize for ghosts
            nextAgent = agent + 1
            if gameState.getNumAgents() == nextAgent:
                nextAgent = 0
            if nextAgent == 0:  #increase depth
                depth += 1
            return min(self.minimax(nextAgent, depth, gameState.generateSuccessor(agent, action)) for action in
                       getLegalActionsNoStop(agent, gameState))

    #function returns minimax action from GameState using self.depth and self.evaluationFunction
    def getAction(self, gameState):
        possibleActions = getLegalActionsNoStop(0, gameState)
        action_scores = [self.minimax(0, 0, gameState.generateSuccessor(0, action)) for action
                         in possibleActions]
        max_action = max(action_scores)
        max_indices = [index for index in range(len(action_scores)) if action_scores[index] == max_action]
        chosenIndex = random.choice(max_indices)
        return possibleActions[chosenIndex]


class GreedyAgent(Agent):
    def __init__(self, evalFn="scoreEvaluation"):
        self.evaluationFunction = util.lookup(evalFn, globals())
        assert self.evaluationFunction != None

    def getAction(self, state):
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        successors = [(state.generateSuccessor(0, action), action)
                      for action in legal]
        scored = [(self.evaluationFunction(state), action)
                  for state, action in successors]
        bestScore = max(scored)[0]
        bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
        return random.choice(bestActions)


#default evaluation function returns the score of the state
def scoreEvaluationFunction(currentGameState):
    return currentGameState.getScore()

#evaluation function takes in the current and proposed successor GameStates and returns a number, the higher the number the better
def evaluationFunction(currentGameState, action):
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    #calculate the distance to the nearest food
    newFoodList = np.array(newFood.asList())
    distanceToFood = [util.manhattanDistance(newPos, food) for food in newFoodList]
    min_food_distance = 0
    if len(newFoodList) > 0:
        min_food_distance = distanceToFood[np.argmin(distanceToFood)]

    #calculate the distance to the nearest ghost
    ghostPositions = np.array(successorGameState.getGhostPositions())
    distanceToGhost = [util.manhattanDistance(newPos, ghost) for ghost in ghostPositions]
    min_ghost_distance = 0
    nearestGhostScaredTime = 0
    if len(ghostPositions) > 0:
        min_ghost_distance = distanceToGhost[np.argmin(distanceToGhost)]
        nearestGhostScaredTime = newScaredTimes[np.argmin(distanceToGhost)]
        if min_ghost_distance <= 1 and nearestGhostScaredTime == 0:
            return -999999
            
        if min_ghost_distance <= 1 and nearestGhostScaredTime > 0:
            return 999999

    value = successorGameState.getScore() - min_food_distance
    if nearestGhostScaredTime > 0:
        value -= min_ghost_distance
    else:
        value += min_ghost_distance
    return value

def scoreEvaluation(state):
    return state.getScore()


class AlphaBetaAgent(MultiAgentSearchAgent):

    def alphabeta(self, agent, depth, gameState, alpha, beta):
        if gameState.isLose() or gameState.isWin() or depth == self.depth:
            return self.evaluationFunction(gameState)
        if agent == 0:  #maximize for pacman
            value = -999999
            for action in getLegalActionsNoStop(agent, gameState):
                value = max(value, self.alphabeta(1, depth, gameState.generateSuccessor(agent, action), alpha, beta))
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return value
        else:   #minimize for ghosts
            nextAgent = agent + 1
            if gameState.getNumAgents() == nextAgent:
                nextAgent = 0
            if nextAgent == 0:
                depth += 1
            for action in getLegalActionsNoStop(agent, gameState):
                value = 999999
                value = min(value, self.alphabeta(nextAgent, depth, gameState.generateSuccessor(agent, action), alpha, beta))
                beta = min(beta, value)
                if beta <= alpha:   #alphaBeta pruning
                    break
            return value

    #function returns minimax action from GameState using self.depth and self.evaluationFunction using alphaBeta pruning
    def getAction(self, gameState):
        possibleActions = getLegalActionsNoStop(0, gameState)
        alpha = -999999
        beta = 999999
        action_scores = [self.alphabeta(0, 0, gameState.generateSuccessor(0, action), alpha, beta) for action
                         in possibleActions]
        max_action = max(action_scores)
        max_indices = [index for index in range(len(action_scores)) if action_scores[index] == max_action]
        chosenIndex = random.choice(max_indices)
        return possibleActions[chosenIndex]
