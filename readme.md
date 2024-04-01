In order to run a single single instance use _DQN.py_
- Replay buffer and target network may be turned off with _--replay_buffer=False --target_net=False_
- Action selection policy may be changed with _--as_policy={ann_egreedy,egreedy,softmax,ann_softmax}_ (default variant uses annealing egreedy)
- Number of train steps may be changed with option _n_
- At the end of the train session an episode with graphical UI will play
- The reward from the live episode as well as the final evaluation reward will be printed

To run the entire experiment suit use _experiment.py_
- Either delete the file _data.json_ or add the argument _overwrite=True_ to the _experiment_ call
- To rerun specific instances modify the _runs_kwargs_ datastructure
- The full experiment suite runs in about 33 minutes on our systems

To produce the report plots run _plotting.py_
- For convenience we have included the data file _data.json_ used in our report, such that the plotting source can be tested directly without rerunning experiment
- The plots will be found as pdfs in the root of the directory
