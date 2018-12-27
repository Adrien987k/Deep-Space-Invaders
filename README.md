Deep-Space-Invaders
===================

Installing the environment
------------ 

Windows :
~~~~bash
pip install gym
pip install --no-index -f https://github.com/Kojoley/atari-py/releases atari_py
pip install gym-retro
~~~~

Linux :
~~~~bash
pip install gym
pip install -e '.[atari]'
pip install gym-retro
~~~~

Then go in the folder 'Roms' and :
~~~~bash
python -m retro.import .
~~~~

