from yaaf.agents import RandomAgent
import numpy as np
from overcooked2 import Overcooked, SingleAgentWrapper, LAYOUTS
from teammates.Astro import AstroHandcoded, AstroSmart
import pickle

STATE_MAP = np.array([
["2", "2", "2", "2", "2", "2", "2", "4", "7", "7", "7", "7", "7", "7", "7"],
["2", "2", "2", "2", "2", "2", "2", "4", "7", "7", "7", "7", "7", "7", "7"],
["2", "2", "2", "2", "2", "2", "2", "4", "4", "4", "7", "7", "7", "7", "7"],
["2", "2", "2", "2", "2", "2", "2", "4", "4", "4", "7", "7", "7", "7", "7"],
["2", "2", "2", "2", "2", "3", "3", "4", "4", "4", "4", "4", "7", "7", "7"],
["2", "2", "2", "2", "2", "3", "3", "4", "4", "4", "4", "4", "4", "7", "7"],
["1", "1", "1", "1", "3", "3", "3", "4", "4", "4", "4", "4", "4", "7", "7"],
["1", "1", "1", "1", "3", "3", "3", "4", "4", "4", "4", "4", "4", "7", "7"],
["1", "1", "1", "1", "3", "3", "3", "4", "4", "4", "4", "4", "4", "7", "7"],
["1", "1", "1", "1", "3", "3", "3", "4", "4", "6", "4", "4", "6", "6", "6"],
["1", "1", "1", "1", "1", "3", "3", "3", "4", "6", "6", "6", "6", "6", "6"],
["0", "0", "0", "0", "0", "5", "5", "5", "5", "5", "5", "5", "5", "5", "5"],
["0", "0", "0", "0", "0", "5", "5", "5", "5", "5", "5", "5", "5", "5", "5"],
["0", "0", "0", "0", "0", "5", "5", "5", "5", "5", "5", "5", "5", "5", "5"],
["0", "0", "0", "0", "0", "5", "5", "5", "5", "5", "5", "5", "5", "5", "5"]])

ACTION_MEANINGS = [
    "move to lower-index node",
    "move to second-lower-index node",
    "move to third-lower-index node",
    "move to fourth-lower-index node",
    "stay",
    "act"
]
S_COEF = 1

ADJACENCY_MATRIX = np.array(
    [
        [0, 1, 0, 0, 0, 1, 0, 0],
        [1, 0, 1, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 1, 1],
        [1, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 1, 0, 1],
        [0, 0, 0, 0, 1, 0, 1, 0],
    ]
)

f = open("output.txt", 'w')

def find_node(state_mdp, action_mdp):
    
    if action_mdp == 4:
        return state_mdp[1]
    
    adjacencies = np.where(ADJACENCY_MATRIX[state_mdp[1]] == 1)[0]

    downgrade_to_lower_index = int(action_mdp) >= len(adjacencies)
    action_mdp = 0 if downgrade_to_lower_index else action_mdp
    node = adjacencies[action_mdp]

    return node

if __name__ == '__main__':

    single_agent = True
    render = True
    render_mode = "plt"  # Available: window (pop-up) and matplotlib (plt.imshow). Video rendering planned for the future.
    num_actions = 4 # Actions to move to x-lower index node (all except "stay" and "act")
    num_nodes = 8
    layout = "Lab"
    

    if single_agent:
        # Option 1
        env = Overcooked(layout=layout)
        agent = AstroSmart(LAYOUTS[layout], 0, env=env)
        teammate = AstroSmart(LAYOUTS[layout], 1, env=env) # 0 - selects Human; 1 - selects Robot 
        env = SingleAgentWrapper(env, teammate)
        transitions = np.zeros((num_actions, num_nodes, num_nodes))
        state = env.reset()
        terminal = False
        scene = LAYOUTS[layout]
        env.env.state[2] = 1
        env.env.state[3] = 13
        env.render(render_mode)

        for h in range(0,4):
            for ac in range(0, num_actions):
                for a in range(0,len(STATE_MAP)):
                    for b in range(0,len(STATE_MAP[a])):
                        if scene[a,b] == " ":
                            env.env.state[0] = a
                            env.env.state[1] = b
                        else:
                            continue
                        
                        state_mdp = agent.state_converter(env.env.state[:9])
                        target_node = find_node(state_mdp, ac)
                        action = agent.action_converter(env.env.state, state_mdp, ac)
                        env.env.state[4] = h
                        print("HEADING: ", env.env.state[4], file=f) 
                        print("state env: ", env.env.state[:6], "state mdp: ", state_mdp, " action:", ACTION_MEANINGS[ac], " target node: ", target_node, file = f) 
                        print("state env: ", env.env.state[:6], "state mdp: ", state_mdp, " action:", ACTION_MEANINGS[ac], " target node: ", target_node)                         
                        next_state, reward, terminal, info = env.step(action)
                        next_state_mdp = agent.state_converter(next_state[:9])
                        print("------- next state: ", next_state[:6], next_state_mdp[1], file=f)
                        if state_mdp[1] == 5 or state_mdp[1] == 6: #SLIPPERY ZONE
                            transitions[ac, state_mdp[1], next_state_mdp[1]] += S_COEF
                            transitions[ac, state_mdp[1], state_mdp[1]] += (1-S_COEF)
                        else: 
                            transitions[ac, state_mdp[1], next_state_mdp[1]] += 1                  
        print(transitions)
        for ac in range(0, num_actions):
            for x in range(0,num_nodes):
                if transitions[ac,x].sum() != 0:
                    transitions[ac,x,:] = transitions[ac, x,:]/transitions[ac,x].sum()
                else:
                    transitions[ac,x,:] = transitions[ac, x,:]

        for x in range(0,num_nodes):
            for y in range(0,num_nodes):
                if np.all((transitions[:,x,y] == 0)):
                    print("Transition from node ", x, " to node ", y, " is not possible.")

        with open("/home/anavc/Simulator/simulator/decision/transitions_human.pickle", "wb") as a:
                    pickle.dump(transitions, a)

        f.close()

    else:
        # Option 2
        env = Overcooked()
        agent = RandomAgent(env.num_joint_actions)
        state = env.reset()
        env.render(render_mode)
        terminal = False
        while not terminal:
            joint_action = agent.action(state)
            next_state, reward, terminal, info = env.step(joint_action)
            env.render(render_mode)


