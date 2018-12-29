Deep-Space-Invaders
===================

Installing the environment
------------ 

Windows :
~~~~bash
pip3 install gym
pip3 install --no-index -f https://github.com/Kojoley/atari-py/releases atari_py
pip3 install gym-retro
~~~~

(may requires certain access rights)
Linux :
~~~~bash
pip3 install gym
pip3 install -e 'gym[atari]'
pip3 install gym-retro
~~~~

Then go in the folder 'Roms' and :
~~~~bash
python3 -m retro.import .
~~~~

