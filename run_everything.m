%% run_everything.m
% Single-file master: create env, load/train agent, run baseline, evaluate, plot, save.
% Place in same folder as HomeEnergyEnv.m. Run by typing `run_everything`.

clear; close all; clc;
rng(0);

%% ---------------- Config ----------------
trainedAgentFile = 'trainedDQN_home.mat';
doTraining = false;        % set true to force training (can take long)
quickTrainEpisodes = 30;   % used only if no saved agent and doTraining==false
numEval = 2;               % how many episodes to evaluate & plot
saveResults = true;

%% ---------------- Create environment ----------------
fprintf('Creating environment...\n');
env = HomeEnergyEnv();
dt_hours = env.Ts / 3600;    % hours per step

%% ---------------- Load or create agent ----------------
if exist(trainedAgentFile,'file') && ~doTraining
    fprintf('Loading trained agent from %s\n', trainedAgentFile);
    S = load(trainedAgentFile);
    if isfield(S,'agent'), agent = S.agent;
    else error('File exists but does not contain variable ''agent''.'); end
else
    % Build Q-network representation robustly (works across many MATLAB versions)
    obsInfo = getObservationInfo(env);
    actInfo = getActionInfo(env);
    numObs = obsInfo.Dimension(1);
    numActions = numel(actInfo.Elements);

    layers = [
        featureInputLayer(numObs,'Normalization','none','Name','state')
        fullyConnectedLayer(128,'Name','fc1')
        reluLayer('Name','r1')
        fullyConnectedLayer(128,'Name','fc2')
        reluLayer('Name','r2')
        fullyConnectedLayer(numActions,'Name','qout')];

    lgraph = layerGraph(layers);
    try
        dlnet = dlnetwork(lgraph); qNet = dlnet;
    catch
        qNet = lgraph;
    end

    try
        qRepresentation = rlQValueRepresentation(qNet, obsInfo, actInfo, 'Observation','state');
    catch
        qRepresentation = rlQValueRepresentation(qNet, obsInfo, actInfo);
    end

    % Exploration object (robust)
    try
        exploration = rl.option.EpsilonGreedyExploration('Epsilon',1.0,'EpsilonMin',0.02,'EpsilonDecay',1e-4);
    catch
        exploration = struct('Epsilon',1.0,'EpsilonMin',0.02,'EpsilonDecay',1e-4); % fallback (older toolbox may ignore)
    end

    agentOpts = rlDQNAgentOptions(...
        'UseDoubleDQN',true, ...
        'TargetSmoothFactor',1e-3, ...
        'ExperienceBufferLength',1e5, ...
        'MiniBatchSize',64, ...
        'DiscountFactor',0.99);

    % attach exploration if class supported
    if isa(exploration,'rl.option.EpsilonGreedyExploration')
        agentOpts.EpsilonGreedyExploration = exploration;
    end

    agent = rlDQNAgent(qRepresentation, agentOpts);

    if doTraining
        % Full training (may be long) - adjust options as you need
        fprintf('Starting full training... (set doTraining=false to skip)\n');
        trainOpts = rlTrainingOptions('MaxEpisodes',800,'MaxStepsPerEpisode',env.MinutesPerEpisode,...
            'Verbose',true,'Plots','training-progress','ScoreAveragingWindowLength',20);
        train(agent, env, trainOpts);
        save(trainedAgentFile,'agent');
    else
        % Quick training so we have a working agent (short, for demo)
        fprintf('No saved agent found or forced training disabled: running short quick training (%d eps)...\n', quickTrainEpisodes);
        trainOpts = rlTrainingOptions('MaxEpisodes',quickTrainEpisodes,'MaxStepsPerEpisode',min(200,env.MinutesPerEpisode),...
            'Verbose',true,'Plots','training-progress','ScoreAveragingWindowLength',10);
        train(agent, env, trainOpts);
        save(trainedAgentFile,'agent');
        fprintf('Quick training completed and saved to %s\n', trainedAgentFile);
    end
end

%% ---------------- Baseline (script-like inline) ----------------
% We'll implement baseline as a function here to avoid workspace confusion.
function [total_cost, avg_comf, ts] = baseline_run_local(env_in)
    dt_h = env_in.Ts/3600;
    obs = reset(env_in);
    done = false;
    total_cost = 0; comfort_sum = 0; t=0;
    temps=[]; outs=[]; socs=[]; prices=[]; gridp=[]; energy=[]; appl=[];
    while ~done
        hour = mod(t/60,24);
        % HVAC: comfort 7-22 else ECO (indices 1/2)
        if hour >=7 && hour<22, HVAC_index = 2; else HVAC_index = 1; end
        % Batt: charge 0-6, discharge 16-20, hold otherwise
        if hour >=0 && hour < 6, Batt_index = 2;
        elseif hour>=16 && hour<20, Batt_index = 0;
        else Batt_index = 1; end
        % appliance: run in 0-6 if not running
        appliance_remain = double(obs(7));
        if appliance_remain <= 0 && hour >=0 && hour < 6, Appl_index = 1; else Appl_index = 0; end

        % decode for power estimate
        HVAC_power = env_in.HVAC_power_levels(HVAC_index+1);
        if Batt_index==0, batt_cmd=-env_in.P_batt_max;
        elseif Batt_index==1, batt_cmd=0;
        else batt_cmd=env_in.P_batt_max; end
        if Appl_index==1 && appliance_remain<=0
            app_now = env_in.appliance_power;
        else
            app_now = (appliance_remain>0)*env_in.appliance_power;
        end
        base_load = 0.6;
        building_power = base_load + HVAC_power;
        grid_power_est = building_power + app_now - max(0,-batt_cmd);
        if batt_cmd>0, grid_power_est = grid_power_est + batt_cmd; end

        a = int32(HVAC_index*6 + Batt_index*2 + Appl_index);
        [nextObs, reward, done, ~] = step(env_in, a);

        temps(end+1,1)=double(obs(1));
        outs(end+1,1)=double(obs(2));
        socs(end+1,1)=double(obs(3));
        prices(end+1,1)=double(obs(6));
        gridp(end+1,1)=grid_power_est;
        energy(end+1,1)=grid_power_est * dt_h * double(obs(6));
        appl(end+1,1)=app_now>0;

        total_cost = total_cost - reward;
        comfort_sum = comfort_sum + abs(obs(1)-env_in.T_pref);

        obs = nextObs;
        t = t + 1;
    end
    avg_comf = comfort_sum / max(1,t);
    ts.temps=temps; ts.outs=outs; ts.socs=socs; ts.prices=prices; ts.grid=gridp; ts.energy=energy; ts.appl=appl; ts.tsteps=(0:length(temps)-1)';
end

fprintf('Running baseline once for summary...\n');
[baseline_cost, baseline_comfort, baseline_ts] = baseline_run_local(env);
fprintf('Baseline complete: cost=%.3f, comfort=%.3f\n', baseline_cost, baseline_comfort);

%% ---------------- Evaluation: run agent vs baseline and collect time-series ----------------
all_agent_costs = zeros(numEval,1);
all_agent_comfort = zeros(numEval,1);
all_baseline_costs = zeros(numEval,1);
all_baseline_comfort = zeros(numEval,1);

last_agent_ts = []; last_baseline_ts = [];

for ep = 1:numEval
    fprintf('\n---- Eval Episode %d/%d ----\n', ep, numEval);
    % Agent rollout
    obs = reset(env); done=false; t=0;
    temps=[]; outs=[]; socs=[]; prices=[]; gridp=[]; energy=[]; appl=[];
    total_cost=0; comfort_sum=0;
    while ~done
        a_raw = getAction(agent, obs);
        a = unwrapAction(a_raw); a = int32(a);
        % decode and estimate grid power pre-step
        HVAC = floor(double(a)/6); remv = mod(double(a),6);
        Batt = floor(remv/2); Appl = mod(remv,2);
        HVAC_power = env.HVAC_power_levels(HVAC+1);
        if Batt==0, batt=-env.P_batt_max; elseif Batt==1, batt=0; else batt=env.P_batt_max; end
        remain = double(obs(7));
        if Appl==1 && remain<=0, app_now=env.appliance_power; else app_now=(remain>0)*env.appliance_power; end
        base_load=0.6;
        grid_est = base_load + HVAC_power + app_now;
        if batt>=0, grid_est = grid_est + batt; else grid_est = grid_est - max(0,-batt); end

        [nextObs, reward, done, ~] = step(env, a);

        temps(end+1,1)=double(obs(1));
        outs(end+1,1)=double(obs(2));
        socs(end+1,1)=double(obs(3));
        prices(end+1,1)=double(obs(6));
        gridp(end+1,1)=grid_est;
        energy(end+1,1)=grid_est * dt_hours * double(obs(6));
        appl(end+1,1)=app_now>0;

        total_cost = total_cost - reward;
        comfort_sum = comfort_sum + abs(obs(1)-env.T_pref);

        obs = nextObs; t = t + 1;
    end
    all_agent_costs(ep)=total_cost;
    all_agent_comfort(ep)=comfort_sum / max(1,t);
    last_agent_ts.temps=temps; last_agent_ts.outs=outs; last_agent_ts.socs=socs; last_agent_ts.prices=prices;
    last_agent_ts.grid=gridp; last_agent_ts.energy=energy; last_agent_ts.appl=appl; last_agent_ts.tsteps=(0:length(temps)-1)';

    % Baseline rollout (fresh env reset inside)
    [b_cost,b_comf,b_ts] = baseline_run_local(env);
    all_baseline_costs(ep)=b_cost;
    all_baseline_comfort(ep)=b_comf;
    last_baseline_ts = b_ts;

    fprintf('Episode %d: Agent cost=%.3f, Baseline cost=%.3f\n', ep, total_cost, b_cost);
end

%% ---------------- Plot results (last episode) ----------------
minutes = last_agent_ts.tsteps;
hours = minutes / 60;

figure('Name','Agent vs Baseline - Last Episode','Position',[200 200 1200 700]);
subplot(3,2,1);
plot(hours, last_agent_ts.temps,'b', hours, last_baseline_ts.temps,'r'); hold on;
yline(env.T_pref,'k--','T_{pref}');
xlabel('Hours'); ylabel('Indoor Temp (°C)'); legend('Agent','Baseline','T_{pref}'); title('Indoor Temperature');
subplot(3,2,2);
plot(hours,last_agent_ts.socs,'b',hours,last_baseline_ts.socs,'r'); xlabel('Hours'); ylabel('SOC'); legend('Agent','Baseline'); title('SOC');
subplot(3,2,3);
plot(hours,last_agent_ts.grid,'b',hours,last_baseline_ts.grid,'r'); xlabel('Hours'); ylabel('Grid Power (kW)'); legend('Agent','Baseline'); title('Grid Power');
subplot(3,2,4);
plot(hours,last_agent_ts.prices,'b',hours,last_baseline_ts.prices,'r'); xlabel('Hours'); ylabel('Price ($/kWh)'); legend('Agent','Baseline'); title('Price');
subplot(3,2,5);
plot(hours,cumsum(last_agent_ts.energy),'b',hours,cumsum(last_baseline_ts.energy),'r'); xlabel('Hours'); ylabel('Cumulative Cost ($)'); legend('Agent','Baseline'); title('Cumulative Cost');
subplot(3,2,6);
stairs(hours,last_agent_ts.appl,'b'); hold on; stairs(hours,last_baseline_ts.appl,'r'); xlabel('Hours'); ylabel('Appliance'); legend('Agent','Baseline'); title('Appliance Status');

figure('Name','Hourly energy cost (last episode)');
numHours = ceil(length(last_agent_ts.energy)/60);
agent_hourly = zeros(numHours,1); base_hourly = zeros(numHours,1);
for h=1:numHours
    idx = ((h-1)*60+1):min(h*60,length(last_agent_ts.energy));
    agent_hourly(h)=sum(last_agent_ts.energy(idx));
    base_hourly(h)=sum(last_baseline_ts.energy(idx));
end
bar([agent_hourly, base_hourly]); xlabel('Hour'); ylabel('Energy cost ($)'); legend('Agent','Baseline'); title('Hourly energy cost');

%% ---------------- Save results ----------------
if saveResults
    save('run_everything_results.mat','all_agent_costs','all_agent_comfort','all_baseline_costs','all_baseline_comfort',...
        'last_agent_ts','last_baseline_ts');
    fprintf('Saved run_everything_results.mat\n');
end

fprintf('\n=== FINISHED: summary ===\n');
fprintf('Agent mean cost = %.3f, Baseline mean cost = %.3f\n', mean(all_agent_costs), mean(all_baseline_costs));
fprintf('Agent mean comfort = %.3f, Baseline mean comfort = %.3f\n', mean(all_agent_comfort), mean(all_baseline_comfort));

%% ---------------- Helper: unwrapAction ----------------
function val = unwrapAction(val)
    if iscell(val) && numel(val)==1, val = val{1}; end
    if isstruct(val) && isfield(val,'Data'), val = val.Data; end
    if isobject(val)
        try
            if isprop(val,'Data'), val = val.Data; end
        catch
        end
    end
    if isstring(val) || ischar(val), val = double(val); end
end

% end of run_everything.m
