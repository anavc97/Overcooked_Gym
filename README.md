# Overcooked_Gym
Overcooked Environment used for the HOTSPOT project


### 
VISUALS
###

- state_visualizer.py: 
- layout_generator.py: 

Alterações:
- alterações nos png (terrain.png, terrain.json)
- alterações no player (se player == x - astro.png, else chef.png)
- Terreno de gelo - se astro está lá, entao escorrega com prob. = 1-S_COEF

#####
AGENT 
#####

- Astro.py
- mdp_ind_[layout_name].npy
- policy_[layout_name].npy


Adições:
- Astro politica otima p/ equipa + se pessoa tem bola na mão, ele vai ter com ela

###
GAME MECHANICS
###

Alterações: 
- Astro não pode apanhar bola
- Se pessoa interage com Astro com bola, bola desaparece
- Jogo acaba quando todas as bolas forem recolhidas
