#!/bin/bash

cd ~/w/
./technical_utils/ruby/markdown_auto_toc/summerizeMD ~/w/nn_samples/draft/DDPG.md -o ~/w/nn_samples/notes/DRL/DDPG.md
./technical_utils/ruby/markdown_auto_toc/summerizeMD ~/w/nn_samples/draft/DRL-Fundamentals.md -o ~/w/nn_samples/notes/DRL/DRL-Fundamentals.md
./technical_utils/ruby/markdown_auto_toc/summerizeMD ~/w/nn_samples/draft/DQN.md -o ~/w/nn_samples/notes/DRL/DQN.md
./technical_utils/ruby/markdown_auto_toc/summerizeMD ~/w/nn_samples/draft/Policy-Gradients.md -o ~/w/nn_samples/notes/DRL/Policy-Gradients.md
./technical_utils/ruby/markdown_auto_toc/summerizeMD ~/w/nn_samples/draft/Terminology.md -o ~/w/nn_samples/notes/Concepts/Terminology.md
./technical_utils/ruby/markdown_auto_toc/summerizeMD ~/w/nn_samples/draft/Q-Learning.md -o ~/w/nn_samples/notes/RL/Q-Learning.md
