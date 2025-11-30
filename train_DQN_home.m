%% train_DQN_home.m
% Training script for DQN on the HomeEnergyEnv (software-only)
% Requires: HomeEnergyEnv.m (class in same folder) and Reinforcement Learning Toolbox.

close all;
clear;
clc;

rng(0); % for reproducibility

%% Create environment
env = HomeEnergyEnv();

obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

numObs = obsInfo.Dimension(1);
actList = actInfo.Elements;
numActions = numel(actList);

%% Build Q-network (MLP)
% Simple MLP mapping observation -> Q-values for each discrete action
layers = [
    featureInputLayer(numObs,'Normalization','none','Name','state')
    fullyConnectedLayer(128,'Name','fc1')
    reluLayer('Name','r1')
    fullyConnectedLayer(128,'Name','fc2')
    reluLayer('Name','r2')
    fullyConnectedLayer(numActions,'Name','qout')];

lgraph = layerGraph(layers);

% Create Q-value representation
% 'Observation' name must match the featureInputLayer name ('state')
qRepresentation = rlQValueRepresentation(lgraph, obsInfo, actInfo, ...
    'Observation','state');

%% Create exploration object (correct class)
% IMPORTANT: use rl.option.EpsilonGreedyExploration object, not a struct.
exploration = rl.option.EpsilonGreedyExploration(...
    'Epsilon',      1.0, ...
    'EpsilonMin',   0.02, ...
    'EpsilonDecay', 1e-4);  % tune decay to your total training steps

%% Create DQN agent options with exploration object
agentOpts = rlDQNAgentOptions(...
    'UseDoubleDQN', true, ...
    'TargetSmoothFactor', 1e-3, ...
    'ExperienceBufferLength', 1e6, ...
    'MiniBatchSize', 256, ...
    'DiscountFactor', 0.99, ...
    'EpsilonGreedyExploration', exploration ...
    );

%% Create the DQN agent
agent = rlDQNAgent(qRepresentation, agentOpts);

%% Training options
maxEpisodes = 800;                       % increase if you need longer training
stepsPerEpisode = env.MinutesPerEpisode; % 24h at 1-min timestep by default

trainOpts = rlTrainingOptions(...
    'MaxEpisodes', maxEpisodes, ...
    'MaxStepsPerEpisode', stepsPerEpisode, ...
    'ScoreAveragingWindowLength', 20, ...
    'Verbose', true, ...
    'Plots','training-progress', ...
    'StopOnError','on');

%% Train agent
doTraining = true;
if doTraining
    fprintf('Starting training: maxEpisodes=%d, stepsPerEpisode=%d\n', maxEpisodes, stepsPerEpisode);
    trainingStats = train(agent, env, trainOpts);
    save('trainedDQN_home.mat','agent','trainingStats');
    fprintf('Training finished and saved to trainedDQN_home.mat\n');
else
    if exist('trainedDQN_home.mat','file')
        load('trainedDQN_home.mat','agent');
        fprintf('Loaded agent from trainedDQN_home.mat\n');
    else
        error('No saved agent found. Set doTraining = true to train.');
    end
end
