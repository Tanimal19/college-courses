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
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # Calculate the distance to the nearest food
        foodList = newFood.asList()
        if foodList:
            minFoodDistance = min(
                [manhattanDistance(newPos, food) for food in foodList]
            )
        else:
            minFoodDistance = 0

        # Calculate the distance to the nearest ghost
        ghostDistances = [
            manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates
        ]
        minGhostDistance = min(ghostDistances)

        # Calculate a score based on the state
        score = successorGameState.getScore()

        # Reward for getting closer to food
        if minFoodDistance > 0:
            score += 1.0 / minFoodDistance

        # Penalize for getting closer to a ghost, unless it's scared
        for ghostState in newGhostStates:
            if ghostState.scaredTimer > 0:
                score += 10  # Reward for being near a scared ghost
            elif minGhostDistance > 0:
                score -= 1.0 / minGhostDistance  # Penalize for being near a ghost

        # Reward for having less food remaining
        score -= len(foodList)

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

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
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

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """

        def minimax(gameState, agentIndex, depth):
            """
            compute the minimax value.
            """
            # Base case: terminal state or maximum depth reached
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)

            # Pacman's turn (maximizing agent)
            if agentIndex == 0:
                legalActions = gameState.getLegalActions(agentIndex)
                if not legalActions:
                    return self.evaluationFunction(gameState)

                max_value = float("-inf")
                for action in legalActions:
                    successor = gameState.generateSuccessor(agentIndex, action)
                    value = minimax(successor, agentIndex + 1, depth)
                    max_value = max(max_value, value)
                return max_value

            # Ghosts' turn (minimizing agents)
            else:
                legalActions = gameState.getLegalActions(agentIndex)
                if not legalActions:
                    return self.evaluationFunction(gameState)

                min_value = float("inf")
                for action in legalActions:
                    successor = gameState.generateSuccessor(agentIndex, action)
                    # If it's the last ghost, decrement the depth
                    if agentIndex == gameState.getNumAgents() - 1:
                        value = minimax(successor, 0, depth - 1)
                    else:
                        value = minimax(successor, agentIndex + 1, depth)
                    min_value = min(min_value, value)
                return min_value

        # Get legal actions for Pacman
        legalActions = gameState.getLegalActions(0)
        if not legalActions:
            return None

        # Choose the action with the highest minimax value
        bestAction = None
        bestValue = float("-inf")
        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            value = minimax(successor, 1, self.depth)  # Start with the first ghost
            if value > bestValue:
                bestValue = value
                bestAction = action

        return bestAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        def value(gameState, agentIndex, depth, alpha, beta):
            """
            Compute the value of a game state.
            """
            # Base case: terminal state or maximum depth reached
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)

            # Pacman's turn (maximizing agent)
            if agentIndex == 0:
                return maxValue(gameState, agentIndex, depth, alpha, beta)

            # Ghosts' turn (minimizing agents)
            else:
                return minValue(gameState, agentIndex, depth, alpha, beta)

        def maxValue(gameState, agentIndex, depth, alpha, beta):
            """
            Compute the max value of a game state.
            """
            v = float("-inf")
            legalActions = gameState.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(gameState)

            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                # Next agent
                nextAgentIndex = agentIndex + 1
                if nextAgentIndex == gameState.getNumAgents():
                    nextAgentIndex = 0
                    nextDepth = depth - 1
                else:
                    nextDepth = depth

                v = max(v, value(successor, nextAgentIndex, nextDepth, alpha, beta))
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v

        def minValue(gameState, agentIndex, depth, alpha, beta):
            """
            Compute the min value of a game state.
            """
            v = float("inf")
            legalActions = gameState.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(gameState)

            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)

                # Next agent
                nextAgentIndex = agentIndex + 1
                if nextAgentIndex == gameState.getNumAgents():
                    nextAgentIndex = 0
                    nextDepth = depth - 1
                else:
                    nextDepth = depth

                v = min(v, value(successor, nextAgentIndex, nextDepth, alpha, beta))
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v

        # Get legal actions for Pacman
        legalActions = gameState.getLegalActions(0)
        if not legalActions:
            return None

        # Choose the action with the highest minimax value
        bestAction = None
        alpha = float("-inf")
        beta = float("inf")
        bestValue = float("-inf")

        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            val = value(successor, 1, self.depth, alpha, beta)

            if val > bestValue:
                bestValue = val
                bestAction = action
            if val > alpha:
                alpha = val

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

        def expectimax(gameState, agentIndex, depth):
            """
            Compute the expectimax value.
            """
            # Base case: terminal state or maximum depth reached
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)

            # Pacman's turn (maximizing agent)
            if agentIndex == 0:
                legalActions = gameState.getLegalActions(agentIndex)
                if not legalActions:
                    return self.evaluationFunction(gameState)

                max_value = float("-inf")
                for action in legalActions:
                    successor = gameState.generateSuccessor(agentIndex, action)
                    value = expectimax(successor, agentIndex + 1, depth)
                    max_value = max(max_value, value)
                return max_value

            # Ghosts' turn (expectimax agents)
            else:
                legalActions = gameState.getLegalActions(agentIndex)
                if not legalActions:
                    return self.evaluationFunction(gameState)

                expected_value = 0
                num_actions = len(legalActions)
                for action in legalActions:
                    successor = gameState.generateSuccessor(agentIndex, action)
                    # If it's the last ghost, decrement the depth
                    if agentIndex == gameState.getNumAgents() - 1:
                        value = expectimax(successor, 0, depth - 1)
                    else:
                        value = expectimax(successor, agentIndex + 1, depth)
                    expected_value += value
                return float(expected_value) / num_actions

        # Get legal actions for Pacman
        legalActions = gameState.getLegalActions(0)
        if not legalActions:
            return None

        # Choose the action with the highest expectimax value
        bestAction = None
        bestValue = float("-inf")
        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            value = expectimax(successor, 1, self.depth)  # Start with the first ghost
            if value > bestValue:
                bestValue = value
                bestAction = action

        return bestAction


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
    This evaluation function encourages Pacman to eat food, hunt scared ghosts, avoid dangerous ghosts,
    and prioritize finishing the game by eating all the food.
    We consider:
    - the current score
    - distance to the nearest food, proximity to ghosts (both scared and not scared)
    - the number of remaining food pellets
    - the distance to the nearest capsule
    """
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    capsules = currentGameState.getCapsules()

    score = currentGameState.getScore()

    # Food score
    foodList = food.asList()
    if foodList:
        minFoodDistance = min([manhattanDistance(pos, food) for food in foodList])
        score += 1.0 / minFoodDistance
    score -= len(foodList) * 2  # Penalize for remaining food

    # Ghost score
    for i, ghostState in enumerate(ghostStates):
        ghostDistance = manhattanDistance(pos, ghostState.getPosition())
        if scaredTimes[i] > 0:
            if ghostDistance > 0:
                score += 2.0 / ghostDistance  # Encourage hunting scared ghosts
        else:
            if ghostDistance < 2:
                score -= 500  # Heavily penalize being too close to a non-scared ghost
            elif ghostDistance > 0:
                score -= 1.0 / ghostDistance  # Penalize being near a ghost

    # Capsule score
    if capsules:
        minCapsuleDistance = min(
            [manhattanDistance(pos, capsule) for capsule in capsules]
        )
        score += 1.0 / minCapsuleDistance

    return score


# Abbreviation
better = betterEvaluationFunction
