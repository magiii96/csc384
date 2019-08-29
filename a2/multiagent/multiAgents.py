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
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        h = currentGameState.getWalls().height
        w = currentGameState.getWalls().width

        "*** YOUR CODE HERE ***"
        c_distance = 100000
        n_food = newFood.count()
        
        for newGhost in newGhostStates:
            #if abs(newPos[0] - newGhost.getPosition()[0]) + abs(newPos[1] - newGhost.getPosition()[1]) <= newScaredTimes:
            #   return 0
            if abs(newPos[0] - newGhost.getPosition()[0]) + abs(newPos[1] - newGhost.getPosition()[1]) <= 1:
                return - h - w
    
        if n_food == 0 :
            return 2*h + 2*w
        else:
            for food_p in newFood.asList():
                distance = abs(newPos[0] - food_p[0]) + abs(newPos[1] - food_p[1])
                if distance < c_distance:
                    c_distance = distance

        if currentGameState.getNumFood() - newFood.count() == 1:
            return 2*h + 2*w - c_distance
        return h + w - c_distance
#return -successorGameState.getScore()- h - w

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
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
        """
        "*** YOUR CODE HERE ***"
        turn = self.index
        move, value = self.minimax(gameState, turn)
        return move

    def minimax(self, pos, turn):
        if self.depth == 0:
            return None, self.evaluationFunctione(pos)
        best_move = None
        value = None
        if pos.isWin() or pos.isLose() or turn >= self.depth * pos.getNumAgents():
            return best_move, self.evaluationFunction(pos)
        if turn % pos.getNumAgents() == 0:
            value = - float('inf')
        if turn % pos.getNumAgents() > 0:
            value = float('inf')
        for move in pos.getLegalActions(turn % pos.getNumAgents()):
            nxt_pos = pos.generateSuccessor(turn % pos.getNumAgents(), move)
            nxt_move, nxt_val = self.minimax(nxt_pos, turn + 1)
            if turn % pos.getNumAgents() == 0 and value < nxt_val:
                best_move, value = move, nxt_val
            if turn % pos.getNumAgents() > 0 and value > nxt_val:
                best_move, value = move, nxt_val
        return best_move, value



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        turn = self.index
        move, value = self.alphabeta(gameState, turn, - float('inf'), float('inf'))
        return move

    def alphabeta(self, pos, turn, alpha, beta):
        best_move = None
        value = None
        if pos.isWin() or pos.isLose() or turn >= self.depth * pos.getNumAgents():
            return best_move,self.evaluationFunction(pos)
        if turn % pos.getNumAgents() == 0:
            value =  - float('inf')
        if turn % pos.getNumAgents() > 0:
            value = float('inf')
        for move in pos.getLegalActions(turn % pos.getNumAgents()):
            nxt_pos = pos.generateSuccessor(turn % pos.getNumAgents(), move)
            nxt_move,nxt_val = self.alphabeta(nxt_pos, turn + 1, alpha, beta)
            if turn % pos.getNumAgents() == 0:
                if value < nxt_val:
                    best_move, value = move, nxt_val
                if value >= beta:
                    return best_move, value
                alpha = max(alpha, value)
            if turn % pos.getNumAgents() > 0:
                if value > nxt_val:
                    best_move, value = move, nxt_val
                if value <= alpha:
                    return best_move, value
                beta = min(beta, value)
        return best_move, value


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
        turn = self.index
        move, value = self.expectimax(gameState, turn)
        return move
    
    def expectimax(self, pos, turn):
        best_move = None
        value = None
        if pos.isWin() or pos.isLose() or turn >= self.depth * pos.getNumAgents():
            return best_move, self.evaluationFunction(pos)
        if turn % pos.getNumAgents() == 0:
            value = - float('inf')
        if turn % pos.getNumAgents() > 0 :
            value = 0
        for move in pos.getLegalActions(turn % pos.getNumAgents()):
            nxt_pos = pos.generateSuccessor(turn % pos.getNumAgents(), move)
            nxt_move, nxt_val = self.expectimax(nxt_pos, turn + 1)
            if turn % pos.getNumAgents() == 0 and value < nxt_val:
                best_move, value = move, nxt_val
            if turn % pos.getNumAgents() > 0 :
                value = value + 1.0/float(len(pos.getLegalActions(turn % pos.getNumAgents()))) * nxt_val
        return best_move, value



def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: 
      First consider the positon of ghost, if we can eat ghost right now and also the ghost is closed to pacman, it better to eat ghost first instead of food. If we cannot eat ghost right now, we must try our best to avoid ghost otherwise, we will lose the game!
      When the ghost is not closed to the pacman, we will choose the state based on food. If there is no other food left, we can win game. If there are many food in the map , we let pacman to go the place where there are the most food we can eat. If there are only a small amout food left , choose the closest food to eat.
      
    """
    "*** YOUR CODE HERE ***"
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    GhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]
    h = currentGameState.getWalls().height
    w = currentGameState.getWalls().width
    n_food = food.count()
    c_distance = float('inf')
    avg_distance = 0
    sum_distance = 0
    
    
    for i in range(0, len(GhostStates)):
        if abs(GhostStates[i].getPosition()[0] - pos[0]) + abs(GhostStates[i].getPosition()[1] - pos[1]) <= newScaredTimes[i]:
            return h + w - abs(GhostStates[i].getPosition()[0] - pos[0]) + abs(GhostStates[i].getPosition()[1] - pos[1]) + 10 * currentGameState.getScore() - 100 * n_food
        elif abs(pos[0] - GhostStates[i].getPosition()[0]) + abs(pos[1] - GhostStates[i].getPosition()[1]) <= 1:
            return - h - w - 10 * currentGameState.getScore() - 10 * n_food
    if n_food == 0:
        return h + w + 10 * currentGameState.getScore()

    for food_p in food.asList():
        distance = abs(pos[0] - food_p[0]) + abs(pos[1] - food_p[1])
        sum_distance += distance
        if distance < c_distance:
            c_distance = distance
    avg_distance = float(sum_distance/n_food)
    if n_food > h * w * 0.5:
        return h + w - avg_distance + 10 * currentGameState.getScore() - 100 * n_food
    else:
        return h + w - c_distance + 10 * currentGameState.getScore() - 100 * n_food


# Abbreviation
better = betterEvaluationFunction

