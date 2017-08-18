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
import math

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        if (legalMoves[chosenIndex] == 'Stop'):
            if len(bestIndices) > 1:
                for num in range(0, len(bestIndices) - 1):
                    if num != chosenIndex:
                        return legalMoves[num]
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        #either scared times or distance from ghosts + 1/foodcount

        successorGameState = currentGameState.generatePacmanSuccessor(action)
        if (successorGameState.isWin()):
            return float("inf")
        if (successorGameState.isLose()):
            return float("-inf")
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newCapsules = successorGameState.getCapsules()
        foodCount = currentGameState.data.layout.totalFood + 1;
        distanceToGhosts = 0
        height = newFood.height
        width = newFood.width
        minDist = float("inf")
        allFoodDistanceSum = 0
        for i in range(0,height - 1):
            for j in range(0,width - 1):
                if (newFood.data[j][i]):
                    currDist = calcDistance(newPos[0], newPos[1], j, i)
                    allFoodDistanceSum += currDist
                    if currDist < minDist:
                        minDist = currDist

        minCDist = float("inf")
        for capsule in newCapsules:
            currDist = calcDistance(newPos[0], newPos[1], capsule[0], capsule[1])
            if currDist < minCDist:
                minCDist = currDist
        newGhostStates = successorGameState.getGhostStates()
        for gs in newGhostStates:
            dist = calcDistance(newPos[0], newPos[1], gs.configuration.pos[0], gs.configuration.pos[1])
            if (not gs.scaredTimer):
                if (dist < 2):
                    rv = float("-inf")
                    return rv
                distanceToGhosts += dist
        score = (1/foodCount) + (1/minDist) + .01*distanceToGhosts + successorGameState.getScore()
        return score

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

def calcDistance(x1, y1, x2, y2):
    return math.sqrt((x2-x1)*(x2-x1) + (y2 - y1)*(y2 - y1))

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

    # def getAction(self, gameState):
    #     initialActions = gameState.getLegalActions(0)
    #     depth = (self.depth) * gameState.getNumAgents()
    #     evaluatedNumbers = [self.recursiveMinimaxer(1, depth - 1, game) for game in [gameState.generateSuccessor(0, act) for act in initialActions]]
    #     bestScore = max(evaluatedNumbers)
    #     bestIndices = [index for index in range(len(evaluatedNumbers)) if evaluatedNumbers[index] == bestScore]
    #     chosenIndex = random.choice(bestIndices) # Pick randomly among the best
    #     return initialActions[chosenIndex]
    #
    # def recursiveMinimaxer(self, index, depth, gamestate):
    #     if (gamestate.isLose() or gamestate.isWin()):
    #         return self.evaluationFunction(gamestate)
    #     actions = gamestate.getLegalActions(index);
    #     if (depth == 0):
    #         return self.evaluationFunction(gamestate)
    #     arrayOfPossibleStates = [gamestate.generateSuccessor(index, act) for act in actions]
    #
    #     if (index == 0 % gamestate.getNumAgents()):
    #         # Maximizer
    #         v = float('-inf')
    #         for successor in arrayOfPossibleStates:
    #             v = max(self.recursiveMinimaxer((index + 1) % gamestate.getNumAgents(), depth - 1, successor), v)
    #         return v
    #     else:
    #         #Minimizer
    #         v = float('inf')
    #         for successor in arrayOfPossibleStates:
    #             v = min(self.recursiveMinimaxer((index + 1) % gamestate.getNumAgents(), depth - 1, successor), v)
    #         return v


    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        initialActions = gameState.getLegalActions(0)
        depth = (self.depth) * gameState.getNumAgents()

        v = float('-inf')
        lowKeyAction = initialActions[0]
        for act in initialActions:
            successor = gameState.generateSuccessor(0, act)
            result = self.recursiveMinimaxer((1) % gameState.getNumAgents(), depth - 1, successor)
            if (result > v):
                v = result
                lowKeyAction = act
        return lowKeyAction


    def recursiveMinimaxer(self, index, depth, gamestate):
        if (gamestate.isLose() or gamestate.isWin()):
            return self.evaluationFunction(gamestate)
        actions = gamestate.getLegalActions(index);
        if (depth == 0):
            return self.evaluationFunction(gamestate)
        if (index == 0 % gamestate.getNumAgents()):
            # Maximizer
            v = float('-inf')
            for act in actions:
                successor = gamestate.generateSuccessor(index, act)
                v = max(self.recursiveMinimaxer((index + 1) % gamestate.getNumAgents(), depth - 1, successor),
                        v)
            return v
        else:
            # Minimizer
            v = float('inf')
            for act in actions:
                successor = gamestate.generateSuccessor(index, act)
                v = min(self.recursiveMinimaxer((index + 1) % gamestate.getNumAgents(), depth - 1, successor),
                        v)
            return v

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        initialActions = gameState.getLegalActions(0)
        depth = (self.depth) * gameState.getNumAgents()

        alpha = float('-inf')
        beta = float('inf')
        v = float('-inf')
        lowKeyAction = initialActions[0]
        for act in initialActions:
            successor = gameState.generateSuccessor(0, act)
            result = self.recursiveMinimaxer((1) % gameState.getNumAgents(), depth - 1, successor, alpha, beta)
            if (result > v):
                v = result
                lowKeyAction = act
            if v > beta:
                return v
            alpha = max(alpha, v)
        return lowKeyAction

    def recursiveMinimaxer(self, index, depth, gamestate, alpha, beta):
        if (gamestate.isLose() or gamestate.isWin()):
            return self.evaluationFunction(gamestate)
        actions = gamestate.getLegalActions(index);
        if (depth == 0):
            return self.evaluationFunction(gamestate)
        if (index == 0 % gamestate.getNumAgents()):
            # Maximizer
            v = float('-inf')
            for act in actions:
                successor = gamestate.generateSuccessor(index, act)
                v = max(self.recursiveMinimaxer((index + 1) % gamestate.getNumAgents(), depth - 1, successor, alpha, beta), v)
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v
        else:
            #Minimizer
            v = float('inf')
            for act in actions:
                successor = gamestate.generateSuccessor(index, act)
                v = min(self.recursiveMinimaxer((index + 1) % gamestate.getNumAgents(), depth - 1, successor, alpha, beta), v)
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        initialActions = gameState.getLegalActions(0)
        depth = (self.depth) * gameState.getNumAgents()

        v = float('-inf')
        lowKeyAction = initialActions[0]
        for act in initialActions:
            successor = gameState.generateSuccessor(0, act)
            result = self.recursiveMinimaxer((1) % gameState.getNumAgents(), depth - 1, successor)
            if (result > v):
                v = result
                lowKeyAction = act
        return lowKeyAction

    def recursiveMinimaxer(self, index, depth, gamestate):
        if (gamestate.isLose() or gamestate.isWin()):
            return self.evaluationFunction(gamestate)
        actions = gamestate.getLegalActions(index);
        if (depth == 0):
            return self.evaluationFunction(gamestate)
        if (index == 0 % gamestate.getNumAgents()):
            # Maximizer
            v = float('-inf')
            for act in actions:
                successor = gamestate.generateSuccessor(index, act)
                v = max(self.recursiveMinimaxer((index + 1) % gamestate.getNumAgents(), depth - 1, successor),
                        v)
            return v
        else:
            # Minimizer
            avgArray = []
            for act in actions:
                successor = gamestate.generateSuccessor(index, act)
                avgArray.append(self.recursiveMinimaxer((index + 1) % gamestate.getNumAgents(), depth - 1, successor))
            return averageArray(avgArray)
def averageArray(arr):
    return float(sum(arr))/max(len(arr),1)
def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>



      Consider nearest food
    """
    "*** YOUR CODE HERE ***"

    if currentGameState.isWin():
        return float("inf")
    if currentGameState.isLose():
        return float("-inf")
    newCapsules = currentGameState.getCapsules()
    newPos = currentGameState.getPacmanPosition()
    score = scoreEvaluationFunction(currentGameState)
    newFood = currentGameState.getFood()
    foodPos = newFood.asList()
    height = newFood.height
    width = newFood.width
    distanceToClosestFood = float("inf")
    allFoodDistanceSum = 0
    for i in range(0, height - 1):
        for j in range(0, width - 1):
            if (newFood.data[j][i]):
                currDist = calcDistance(newPos[0], newPos[1], j, i)
                allFoodDistanceSum += currDist
                if currDist < distanceToClosestFood:
                    distanceToClosestFood = currDist

    minCDist = float("inf")
    for capsule in newCapsules:
        currDist = calcDistance(newPos[0], newPos[1], capsule[0], capsule[1])
        if currDist < minCDist:
            minCDist = currDist
    numghosts = currentGameState.getNumAgents() - 1
    i = 1
    nearestGhost = float("inf")
    while i <= numghosts:
        dist = util.manhattanDistance(newPos, currentGameState.getGhostPosition(i))
        nearestGhost = min(nearestGhost, dist)
        i += 1
    # distanceToGhosts = 0
    # newGhostStates = currentGameState.getGhostStates()
    # for gs in newGhostStates:
    #     dist = calcDistance(newPos[0], newPos[1], gs.configuration.pos[0], gs.configuration.pos[1])
    #     if (not gs.scaredTimer):
    #         distanceToGhosts += dist

    return score + max(nearestGhost, 4) - (min(distanceToClosestFood, minCDist) * 1.5) - (4 * len(foodPos)) - (3*len(newCapsules))

# Abbreviation
better = betterEvaluationFunction

