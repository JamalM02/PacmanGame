import random, util
from game import Agent

# ********* Reflex agent- sections a and b *********
class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.
    """
    def __init__(self):
        self.lastPositions = []
        self.dc = None

    def getAction(self, gameState):
        """
        getAction chooses among the best options according to the evaluation function.
        getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
        ------------------------------------------------------------------------------
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current GameState (pacman.py) and the proposed action
        and returns a number, where higher numbers are better.
        """
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        return betterEvaluationFunction(successorGameState)

# ********* Evaluation functions *********
def scoreEvaluationFunction(gameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.
    """
    return gameState.getScore()

# b: implementing a better heuristic function
def betterEvaluationFunction(gameState):
    """
    A better evaluation function for the ReflexAgent.
    """
    # Useful information you can extract from a GameState (pacman.py)
    pacmanPos = gameState.getPacmanPosition()
    food = gameState.getFood()
    ghosts = gameState.getGhostStates()
    capsules = gameState.getCapsules()
    score = gameState.getScore()

    # Calculate the distance to the closest food
    foodDistances = [util.manhattanDistance(pacmanPos, foodPos) for foodPos in food.asList()]
    minFoodDistance = min(foodDistances) if foodDistances else 0

    # Calculate the distance to the closest ghost
    ghostDistances = [util.manhattanDistance(pacmanPos, ghost.getPosition()) for ghost in ghosts]
    minGhostDistance = min(ghostDistances) if ghostDistances else float('inf')

    # Calculate the distance to the closest capsule
    capsuleDistances = [util.manhattanDistance(pacmanPos, cap) for cap in capsules]
    minCapsuleDistance = min(capsuleDistances) if capsuleDistances else 0

    # Avoid ghosts
    ghostPenalty = 1.0 / minGhostDistance if minGhostDistance > 0 else 0

    return score - minFoodDistance + ghostPenalty - minCapsuleDistance



# ********* MultiAgent Search Agents- sections c,d,e,f *********
class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers. Any methods defined here will be available
    to the MinimaxAgent, AlphaBetaAgent & both ExpectimaxAgents.

    You *do not* need to make any changes here, but you can if you want
    to add functionality to all your adversarial search agents. Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated. It's
    only partially specified, and designed to be extended. Agent (game.py)
    is another abstract class.
    """
    def __init__(self, evalFn='betterEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

# c: implementing minimax
class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent
    """
    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction. Terminal states can be found by one of the following:
        pacman won, pacman lost or there are no legal moves.
        """
        def minimax(agentIndex, depth, gameState):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            legalMoves = gameState.getLegalActions(agentIndex)
            if agentIndex == 0:  # Pacman (maximizer)
                return max(minimax(1, depth, gameState.generateSuccessor(agentIndex, action)) for action in legalMoves)
            else:  # Ghosts (minimizers)
                nextAgent = agentIndex + 1 if agentIndex < gameState.getNumAgents() - 1 else 0
                nextDepth = depth + 1 if nextAgent == 0 else depth
                return min(minimax(nextAgent, nextDepth, gameState.generateSuccessor(agentIndex, action)) for action in legalMoves)

        legalMoves = gameState.getLegalActions(0)
        bestMove = max(legalMoves, key=lambda action: minimax(1, 0, gameState.generateSuccessor(0, action)))
        return bestMove

# d: implementing alpha-beta
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning
    """
    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        def alphabeta(agentIndex, depth, gameState, alpha, beta):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            legalMoves = gameState.getLegalActions(agentIndex)
            if agentIndex == 0:  # Pacman (maximizer)
                value = float('-inf')
                for action in legalMoves:
                    value = max(value, alphabeta(1, depth, gameState.generateSuccessor(agentIndex, action), alpha, beta))
                    if value > beta:
                        return value
                    alpha = max(alpha, value)
                return value
            else:  # Ghosts (minimizers)
                value = float('inf')
                nextAgent = agentIndex + 1 if agentIndex < gameState.getNumAgents() - 1 else 0
                nextDepth = depth + 1 if nextAgent == 0 else depth
                for action in legalMoves:
                    value = min(value, alphabeta(nextAgent, nextDepth, gameState.generateSuccessor(agentIndex, action), alpha, beta))
                    if value < alpha:
                        return value
                    beta = min(beta, value)
                return value

        legalMoves = gameState.getLegalActions(0)
        bestMove = max(legalMoves, key=lambda action: alphabeta(1, 0, gameState.generateSuccessor(0, action), float('-inf'), float('inf')))
        return bestMove

# e: implementing random expectimax
class RandomExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent
    """
    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction
        """
        def expectimax(agentIndex, depth, gameState):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            legalMoves = gameState.getLegalActions(agentIndex)
            if agentIndex == 0:  # Pacman (maximizer)
                return max(expectimax(1, depth, gameState.generateSuccessor(agentIndex, action)) for action in legalMoves)
            else:  # Ghosts (expectation)
                nextAgent = agentIndex + 1 if agentIndex < gameState.getNumAgents() - 1 else 0
                nextDepth = depth + 1 if nextAgent == 0 else depth
                return sum(expectimax(nextAgent, nextDepth, gameState.generateSuccessor(agentIndex, action)) for action in legalMoves) / len(legalMoves)

        legalMoves = gameState.getLegalActions(0)
        bestMove = max(legalMoves, key=lambda action: expectimax(1, 0, gameState.generateSuccessor(0, action)))
        return bestMove

# f: implementing directional expectimax
class DirectionalExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent
    """
    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction
        """
        def expectimax(agentIndex, depth, gameState):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            legalMoves = gameState.getLegalActions(agentIndex)
            if agentIndex == 0:  # Pacman (maximizer)
                return max(expectimax(1, depth, gameState.generateSuccessor(agentIndex, action)) for action in legalMoves)
            else:  # Ghosts (expectation)
                nextAgent = agentIndex + 1 if agentIndex < gameState.getNumAgents() - 1 else 0
                nextDepth = depth + 1 if nextAgent == 0 else depth
                directionalGhostProbabilities = self.getDirectionalGhostProbabilities(gameState, agentIndex)
                return sum(expectimax(nextAgent, nextDepth, gameState.generateSuccessor(agentIndex, action)) * prob for action, prob in directionalGhostProbabilities.items())

        legalMoves = gameState.getLegalActions(0)
        bestMove = max(legalMoves, key=lambda action: expectimax(1, 0, gameState.generateSuccessor(0, action)))
        return bestMove

    def getDirectionalGhostProbabilities(self, gameState, ghostIndex):
        """
        This function should return a dictionary where the keys are legal actions and
        the values are the probabilities of those actions based on the ghost's strategy.
        """
        # Example of equal probability distribution
        legalMoves = gameState.getLegalActions(ghostIndex)
        prob = 1.0 / len(legalMoves)
        return {action: prob for action in legalMoves}

# I: implementing competition agent
class CompetitionAgent(MultiAgentSearchAgent):
    """
    Your competition agent
    """
    def getAction(self, gameState):
        """
        Returns the action using self.depth and self.evaluationFunction
        """
        def competitionExpectimax(agentIndex, depth, gameState):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            legalMoves = gameState.getLegalActions(agentIndex)
            if agentIndex == 0:  # Pacman (maximizer)
                return max(competitionExpectimax(1, depth, gameState.generateSuccessor(agentIndex, action)) for action in legalMoves)
            else:  # Ghosts (expectation)
                nextAgent = agentIndex + 1 if agentIndex < gameState.getNumAgents() - 1 else 0
                nextDepth = depth + 1 if nextAgent == 0 else depth
                return sum(competitionExpectimax(nextAgent, nextDepth, gameState.generateSuccessor(agentIndex, action)) for action in legalMoves) / len(legalMoves)

        legalMoves = gameState.getLegalActions(0)
        bestMove = max(legalMoves, key=lambda action: competitionExpectimax(1, 0, gameState.generateSuccessor(0, action)))
        return bestMove
