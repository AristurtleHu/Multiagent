# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        if childGameState.isWin():
            return float('inf')
        if childGameState.isLose():
            return -float('inf')

        score = childGameState.getScore()

        # Prefer states that are closer to the nearest food.
        foodList = newFood.asList()
        if foodList:
            nearestFood = min(manhattanDistance(newPos, food) for food in foodList)
            score += 10.0 / nearestFood

        # Chase scared ghosts, avoid active ones.
        for ghostState, scaredTime in zip(newGhostStates, newScaredTimes):
            dist = manhattanDistance(newPos, ghostState.getPosition())
            if scaredTime > 0:
                score += 30.0 / (dist + 1)
            else:
                score -= 30.0 / (dist + 1)
                if dist <= 2:
                    score -= 500.0

        return score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        def minimax(state, depth, agentIndex):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            if agentIndex == 0:
                return max(minimax(state.getNextState(0, action), depth, 1)
                           for action in state.getLegalActions(0))

            nextAgent = agentIndex + 1
            nextDepth = depth
            if nextAgent == state.getNumAgents():
                nextAgent, nextDepth = 0, depth + 1
            return min(minimax(state.getNextState(agentIndex, action), nextDepth, nextAgent)
                       for action in state.getLegalActions(agentIndex))

        bestAction, bestScore = None, -float('inf')
        for action in gameState.getLegalActions(0):
            score = minimax(gameState.getNextState(0, action), 0, 1)
            if score > bestScore:
                bestAction, bestScore = action, score
        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        def alphabeta(state, depth, agentIndex, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            if agentIndex == 0:
                value = -float('inf')
                for action in state.getLegalActions(0):
                    value = max(value, alphabeta(state.getNextState(0, action),
                                                 depth, 1, alpha, beta))
                    if value > beta:
                        break
                    alpha = max(alpha, value)
                return value

            nextAgent = agentIndex + 1
            nextDepth = depth
            if nextAgent == state.getNumAgents():
                nextAgent, nextDepth = 0, depth + 1
            value = float('inf')
            for action in state.getLegalActions(agentIndex):
                value = min(value, alphabeta(state.getNextState(agentIndex, action),
                                             nextDepth, nextAgent, alpha, beta))
                if value < alpha:
                    break
                beta = min(beta, value)
            return value

        bestAction, bestScore = None, -float('inf')
        alpha = -float('inf')
        for action in gameState.getLegalActions(0):
            score = alphabeta(gameState.getNextState(0, action), 0, 1, alpha, float('inf'))
            if score > bestScore:
                bestAction, bestScore = action, score
            alpha = max(alpha, bestScore)
        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        def expectimax(state, depth, agentIndex):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            actions = state.getLegalActions(agentIndex)
            if agentIndex == 0:
                return max(expectimax(state.getNextState(0, action), depth, 1)
                           for action in actions)

            nextAgent = agentIndex + 1
            nextDepth = depth
            if nextAgent == state.getNumAgents():
                nextAgent, nextDepth = 0, depth + 1
            return sum(expectimax(state.getNextState(agentIndex, action), nextDepth, nextAgent)
                       for action in actions) / len(actions)

        legalActions = gameState.getLegalActions(0)
        scores = [expectimax(gameState.getNextState(0, action), 0, 1)
                  for action in legalActions]
        return legalActions[scores.index(max(scores))]

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: A combination of the current game score, the distance to
    the closest food, and the distance to each ghost (chasing scared ghosts
    and avoiding active ones). Terminal states use a large finite value
    instead of infinity so that immediate wins are preferred over delayed
    ones.
    """
    if currentGameState.isWin():
        return 1e6
    if currentGameState.isLose():
        return -1e6

    pos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()

    score = currentGameState.getScore()

    if foodList:
        score += 10.0 / min(manhattanDistance(pos, food) for food in foodList)

    for ghostState in ghostStates:
        dist = manhattanDistance(pos, ghostState.getPosition())
        if ghostState.scaredTimer > 0:
            score += 200.0 / (dist + 1)
        else:
            score -= 100.0 / (dist + 1)
            if dist <= 2:
                score -= 1000.0

    return score

# Abbreviation
better = betterEvaluationFunction
