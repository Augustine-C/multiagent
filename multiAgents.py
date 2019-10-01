# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util
import pacman

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
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self,
                           currentGameState,  # type: pacman.GameState
                           action):
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

        "*** YOUR CODE HERE ***"
        # return successorGameState.getScore()
        if successorGameState.isLose():
            return float("-inf")  # negative infinity
        elif successorGameState.isWin() or newPos in currentGameState.getCapsules():
            return float("inf")  # positive infinity

        ghost_positions = currentGameState.getGhostPositions()
        ghost_dists = (manhattanDistance(ghost_pos, newPos) for ghost_pos in ghost_positions)
        close_ghost = min(ghost_dists)
        # if len(currentGameState.getCapsules()) > len(successorGameState.getCapsules()):
        #     return float("inf")
        close_food = min(newFood.asList(), key=lambda x: manhattanDistance(newPos, x))
        if close_ghost <= 2:
            return float("-inf")

        score = close_ghost
        if currentGameState.getNumFood() > successorGameState.getNumFood():
            return float("inf")

        score = score - 3 * manhattanDistance(newPos, close_food)

        return score


def scoreEvaluationFunction(currentGameState):
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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
  """

    def getAction(self,
                  gameState  # type: pacman.GameState
                  ):
        """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
        "*** YOUR CODE HERE ***"

        def max_pac(state,  # type: pacman.GameState
                    depth):
            if state.isWin() or state.isLose():
                return state.getScore()
                # return self.evaluationFunction(gameState)
            pacman_actions = state.getLegalActions(0)
            if Directions.STOP in pacman_actions:
                pacman_actions.remove(Directions.STOP)
            successor_states = ([state.generateSuccessor(0, a), a]
                                for a in pacman_actions)
            if depth == 1:
                return max([min_pac(s, depth, 1), a] for s, a in successor_states)  # value, a
            return max(min_pac(s, depth, 1) for s, a in successor_states)  # value, a

        def min_pac(state,  # type: pacman.GameState
                    depth, index):
            if state.isWin() or state.isLose():
                return self.evaluationFunction(gameState)
            actions = state.getLegalActions(index)
            if Directions.STOP in actions:
                actions.remove(Directions.STOP)
            if index == state.getNumAgents() - 1 and depth == self.depth:
                return min(self.evaluationFunction(state.generateSuccessor(index, a)) for a in actions)
            if index == state.getNumAgents() - 1:
                depth += 1
                return min(max_pac(state.generateSuccessor(index, a), depth) for a in actions)
            successor_states = (state.generateSuccessor(index, a) for a in actions)
            return min(min_pac(s, depth, index + 1) for s in successor_states)  # value, a

        # start
        _, action = max_pac(gameState, 1)
        return action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
  """

    def getAction(self, gameState):
        """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
        "*** YOUR CODE HERE ***"

        def max_pac(state,  # type: pacman.GameState
                    depth, alpha, beta):
            if state.isWin() or state.isLose():
                # return self.evaluationFunction(gameState)
                return state.getScore()
            actions = state.getLegalActions(0)
            if Directions.STOP in actions:
                actions.remove(Directions.STOP)
            max_score = float("-inf")
            max_action = None
            for a in actions:
                score = min_pac(state.generateSuccessor(0, a), depth, 1, alpha, beta)
                if score > max_score:
                    max_score = score
                    max_action = a
                alpha = max(alpha, max_score)
                if alpha >= beta:
                    return max_score
            return max_action if depth == 1 else max_score

        def min_pac(state,  # type: pacman.GameState
                    depth, index, alpha, beta):
            if state.isWin() or state.isLose():
                return state.getScore()
                # return self.evaluationFunction(gameState)
            actions = state.getLegalActions(index)
            if Directions.STOP in actions:
                actions.remove(Directions.STOP)
            min_score = float("inf")
            for a in actions:
                if index == state.getNumAgents() - 1 and depth == self.depth:
                    score = self.evaluationFunction(state.generateSuccessor(index, a))
                elif index == state.getNumAgents() - 1:
                    score = max_pac(state.generateSuccessor(index, a), depth + 1, alpha, beta)
                else:
                    score = min_pac(state.generateSuccessor(index, a), depth, index + 1, alpha, beta)
                if score < min_score:
                    min_score = score
                beta = min(beta, score)
                if alpha >= beta:
                    return min_score
            return min_score

        # start
        return max_pac(gameState, 1, float("-inf"), float("inf"))


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
  """

    def getAction(self, gameState):
        """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
        "*** YOUR CODE HERE ***"

        def max_pac(state,  # type: pacman.GameState
                    depth):
            if state.isWin() or state.isLose():
                return self.evaluationFunction(gameState)
            pacman_actions = state.getLegalActions(0)
            if Directions.STOP in pacman_actions:
                pacman_actions.remove(Directions.STOP)

            successor_states = ([state.generateSuccessor(0, a), a]
                                for a in pacman_actions)
            if depth == 1:
                return max([min_pac(s, depth, 1), a] for s, a in successor_states)  # value, a
            return max(min_pac(s, depth, 1) for s, a in successor_states)  # value

        def min_pac(state,  # type: pacman.GameState
                    depth, index):
            if state.isLose():
                return state.getScore()
            actions = state.getLegalActions(index)
            numberAction = len(actions)
            if numberAction == 0:
                return self.evaluationFunction(state)
            if index == state.getNumAgents() - 1 and depth == self.depth:
                return sum(self.evaluationFunction(state.generateSuccessor(index, a)) for a in actions) / numberAction
            if index == state.getNumAgents() - 1:
                depth += 1
                return sum((max_pac(state.generateSuccessor(index, a), depth) for a in actions)) / numberAction
            successor_states = (state.generateSuccessor(index, a) for a in actions)
            return sum(min_pac(s, depth, index + 1) for s in successor_states) / numberAction  # value, a

        # start
        _, action = max_pac(gameState, 1)
        return action




def betterEvaluationFunction(currentGameState # type: pacman.GameState
                             ):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """
    "*** YOUR CODE HERE ***"
    game_score = currentGameState.getScore()
    if currentGameState.isLose():
        return game_score / 2 if game_score > 0 else game_score * 2
    elif currentGameState.isWin():
        return game_score
    foods = currentGameState.getFood().asList()
    ghosts = currentGameState.getGhostPositions()
    man = currentGameState.getPacmanPosition()

    # eval food info
    close_food = min(manhattanDistance(man, f) for f in foods)
    number_food = len(foods)
    close_food = min(manhattanDistance(man, f) for f in foods)
    food_score = 0
    # if close_food < 3:
    #     food_score = -close_food
    food_score += 300. / number_food - close_food * 1.2

    # eval ghost
    ghost_states = currentGameState.getGhostStates()
    scare_times = [[ghostState.scaredTimer,
                    manhattanDistance(man, ghostState.getPosition())]
                   for ghostState in ghost_states]
    ghost_dists = (dis if time < dis else -100. / dis for time, dis in scare_times)
    close_ghost = min(ghost_dists)
    ghost_score = 0

    if close_ghost < 0:
        ghost_score = close_ghost
    else:
        if close_ghost == 1:
            ghost_score += 150
        if close_ghost == 2:
            ghost_score += 60

    # ghost_score = 100. / close_ghost if close_ghost <= 2 else 0 - close_ghost

    # eval caps
    cap_score = 0
    caps = len(currentGameState.getCapsules())
    cap_score = 30 / caps if caps != 0 else 40

    rs = food_score + game_score - ghost_score + cap_score
    # print 'ghost', close_ghost, 300.0/close_ghost, 'food', food_score, 300.0/food_score
    return rs




# Abbreviation
better = betterEvaluationFunction


class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest
  """

    def getAction(self, gameState):
        """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()
