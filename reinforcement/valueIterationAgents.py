# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# The core projects and autograders were primarily created by John DeNero
# The core projects and autograders were primarily created by John DeNero
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).util


import mdp, util


from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:



              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        # Step 1: Initialised the values as 0 for all the possible states
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # Aim: is to update the value of each element 
        # Write value iteration code here
        
        # use a while loop to iterate through the number of iterations
        numIterations = self.iterations
        
        # initialised the index to trace the iteration in the while loop
        iterationIndex = 0

        while iterationIndex < numIterations:
            
            # new dict is initialised where all the states have 0 value
            newStates = util.Counter()
            states = self.mdp.getStates()

            for state in states:

                # maxvalue is initialised for comparison between all the possible q values
                maxQValue = -9999                
                actions = self.mdp.getPossibleActions(state)

                for action in actions:
                    
                    qValue = 0
                    nextStates = self. mdp.getTransitionStatesAndProbs(state, action)
                    
                    for pairs in nextStates:
                        probabilities = pairs[1]
                        nextState = pairs[0]
                        rewards = self.mdp.getReward(state,action,nextState)
                        qValue +=  probabilities * (rewards + self.discount*(self.getValue(nextState)))
                    
                    if qValue > maxQValue:
                        maxQValue = qValue
                    newStates[state] = maxQValue

            self.values = newStates
            iterationIndex+=1

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        # For 100 iterations
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # Q(s,a) = summation(probability((reward from s to s' by taking action a)+discount*(value of next state)))
        # next state can be obtained using the transition function
        Q_value = 0

        StatesAndProbs = self.mdp.getTransitionStatesAndProbs(state,action)

        for pairs in StatesAndProbs:
            
            probabilities = pairs[1]
            nextState = pairs[0]
            rewards = self.mdp.getReward(state,action,nextState)
            Q_value +=  probabilities * (rewards + self.discount*(self.getValue(nextState)))
        
        return Q_value

        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # THe best value could be computed using the q-value function 
        # compute Value = max (q-value)
        # AIM: to find the action associated with that value
        maxQValue = -99999
        qAction = None

        actions = self.mdp.getPossibleActions(state)
        for action in actions:
            qValue = self.computeQValueFromValues(state,action)
            if qValue > maxQValue:
                maxQValue = qValue
                qAction = action

        return qAction
        # would be redundant and is not solving the problem, so
        # for nextState in self.mdp.getTransitionStatesAndProbs(state,actions)[0]:
        #     if bestValue < self.getValue(nextState):
        #         bestValue = self.getValue(nextState)
        #         bestAction = actions

            
            # print("actions: ",actions)
        # util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
         
        # Do we need to initialise everything to 0 first?
        # index is needed for the circular update in teh iterations 
       
        states = self.mdp.getStates()
        statesLen = len(states)
        numIterations = self.iterations
        iterationIndex = 0

        while iterationIndex < numIterations:
            stateIndex = iterationIndex % statesLen
            state = states[stateIndex]
        
            if self.mdp.isTerminal(state) == False:
                legalActions = self.mdp.getPossibleActions(state)
                bestValue = -9999
                for action in legalActions:
                    value = self.computeQValueFromValues(state,action)
                    if bestValue < value:
                        bestValue = value
                    # All the best Values from state to that particular action
                    # computeActionFRomValues will supposedly be used to find the best action for the 
                    # bestValues of all the possible actions

                self.values[state] = bestValue
            iterationIndex+=1
        # for i in self.mdp.getStates():
        #     states.append(i)
        #     values[states[i]] = 0

        # for i in range(0,self.iterations):
        #     oldValues = util.Counter()
        #     bestValue = -9999

        #     for action in self.mdp.getPossibleActions(states[i]):
        #         value = self.computeQValueFromValues(states[i],action)

        #         if value > bestValue:
        #             bestValue = value

        #         oldValues[states[i]] = bestValue

        #     self.values = oldValues
                # # return the best possible q value 
                # bestValue = -9999
                # for action in self.mdp.getPossibleActions(states):
                #     value = self.computeQValueFromValues(states,action)
                #     # print("value: ", value)
                #     if bestValue < value:
                #         bestValue = value
                #     # All the best Values from states to that particular action
                #     # computeActionFRomValues will supposedly be used to find the best action for the 
                #     # bestValues of all the possible actions

            #         oldValues[states] = bestValue
            # self.values = oldValues
        "*** YOUR CODE HERE ***"

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        # Part 1- Finding all the predecessors:
        # for state in self.mdp.getStates():
        priorityQ = util.PriorityQueue()

        # Consider terminal states?
        self.predecessors = set()
        states = self.mdp.getStates()
        for state in states:
            isTerminal = self.mdp.isTerminal(state)
            if isTerminal ==  False: 
                actions = self.mdp.getPossibleActions(state)
                for action in actions:
                    statesAndProbsPairs = self.mdp.getTransitionStatesAndProbs(state, action)
                    for pair in statesAndProbsPairs:
                        # print(pair)
                        if pair[1] > 0:
                            self.predecessors.add(pair[0])
        # print(self.predecessors)

        # Part 2 - Initialise an empty priority queue
        
        states = self.mdp.getStates()
        for state in states:

            diff = 0
            
            isTerminal = self.mdp.isTerminal(state)
            if isTerminal ==  False:

                highestQvalue = -99999
                actions = self.mdp.getPossibleActions(state)

                for action in actions:
                    qValue = self.computeQValueFromValues(state,action)
                    if qValue > highestQvalue:
                       highestQvalue = qValue

                diff = abs(self.getValue(state) - highestQvalue)

            priorityQ.update(state,-diff)
            # print(priorityQ)

        # Part 3 - 
        iterationIndex = 0
        numIterations = self.iterations
        theta = self.theta
        
        while (iterationIndex < numIterations):

            if priorityQ.isEmpty() == True:
                break

            removedState = priorityQ.pop()

            if self.mdp.isTerminal(removedState) == False:
                # Update value to the highest q value
                bestValue = -99999
                actions = self.mdp.getPossibleActions(removedState)

                for action in actions:
                    value = self.computeQValueFromValues(removedState,action)
                    if value > bestValue:
                       bestValue = value

                self.values[removedState] = bestValue

                for pred in self.predecessors:

                    diff2 = 0
                    bestValue2 = -99999
                    actions2 = self.mdp.getPossibleActions(pred)
                    isTerminal = self.mdp.isTerminal(pred)
                    if isTerminal == False:
                        
                        for action in actions2:
                            value2 = self.computeQValueFromValues(pred,action)
                            if value2 > bestValue2:
                               bestValue2 = value2

                        diff2 = abs(self.getValue(pred) - bestValue2)

                        if diff2 > theta:
                            priorityQ.update(pred,-diff2)
                            
            iterationIndex+=1


        # For each non-terminal state, compute the abs difference between current values and the highest q value across all possible actions from s - Diff
        # Push s into the priority qeuee with priority as diff
         
        "*** YOUR CODE HERE ***"

