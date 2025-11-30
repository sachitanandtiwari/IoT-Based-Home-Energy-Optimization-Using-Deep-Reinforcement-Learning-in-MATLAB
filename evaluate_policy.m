%% evaluate_policy_with_plots.m
% Evaluate trained DQN agent vs rule-based baseline and plot time-series.

clear; close all; clc;

%% Config
trainedAgentFile = 'trainedDQN_home.mat';
numEval = 3;       
saveResults = true;

%% Load trained agent
S = load(trainedAgentFile);
agent = S.agent;

%% Create environment
env = HomeEnergyEnv();
dt_hours = env.Ts / 3600;     % timestep in hours

%% Storage for summary metrics
agent_costs = zeros(numEval,1);
agent_comfort = zeros(numEval,1);
baseline_costs = zeros(numEval,1);
baseline_comfort = zeros(numEval,1);

%% Loop over episodes
for ep = 1:numEval
    fprintf("\n=== Episode %d/%d ===\n", ep, numEval);

    %% ------------------- Agent rollout -------------------
    obs = reset(env);
    done = false;

    temps=[]; outs=[]; socs=[]; prices=[]; gridP=[]; energyC=[]; appl=[];

    total_cost = 0;
    comfort_sum = 0;
    step_count = 0;

    while ~done
        % Get action
        a_raw = getAction(agent, obs);
        a = unwrapAction(a_raw);
        a = int32(a);

        % Decode action
        HVAC = floor(a/6); remv = mod(a,6);
        Batt = floor(remv/2);
        Appl = mod(remv,2);

        % HVAC power
        HVAC_power = env.HVAC_power_levels(HVAC+1);

        % battery
        if Batt == 0
            batt_cmd = -env.P_batt_max;
        elseif Batt == 1
            batt_cmd = 0;
        else
            batt_cmd = +env.P_batt_max;
        end

        % appliance
        remain = double(obs(7));
        if Appl == 1 && remain <= 0
            app_now = env.appliance_power;
        elseif remain > 0
            app_now = env.appliance_power;
        else
            app_now = 0;
        end

        % estimated grid power
        base_load = 0.6;
        grid_est = base_load + HVAC_power + app_now;
        if batt_cmd >= 0
            grid_est = grid_est + batt_cmd;
        else
            grid_est = grid_est - max(0,-batt_cmd);
        end

        % Step env
        [nextObs, reward, done] = step(env, a);

        % Log data
        temps(end+1)=obs(1);
        outs(end+1)=obs(2);
        socs(end+1)=obs(3);
        prices(end+1)=obs(6);
        gridP(end+1)=grid_est;
        energyC(end+1)=grid_est * dt_hours * obs(6);
        appl(end+1)=app_now>0;

        total_cost = total_cost - reward;
        comfort_sum = comfort_sum + abs(obs(1) - env.T_pref);

        obs = nextObs;
        step_count = step_count + 1;
    end

    agent_costs(ep) = total_cost;
    agent_comfort(ep) = comfort_sum / max(1,step_count);

    fprintf("Agent: cost=%.3f  comfort=%.3f\n", agent_costs(ep), agent_comfort(ep));

    % Save agent timeseries
    agent_ts.temps = temps;
    agent_ts.outs = outs;
    agent_ts.socs = socs;
    agent_ts.prices = prices;
    agent_ts.grid = gridP;
    agent_ts.energy = energyC;
    agent_ts.appl = appl;
    agent_ts.steps = (0:length(temps)-1)';

    %% ------------------- BASELINE rollout -------------------
    [b_cost, b_comf, b_ts] = baseline_rollout(env);   % <<< FIXED

    baseline_costs(ep) = b_cost;
    baseline_comfort(ep) = b_comf;

    baseline_ts = b_ts;

    fprintf("Baseline: cost=%.3f  comfort=%.3f\n", b_cost, b_comf);
end

%% SUMMARY
fprintf("\n=== SUMMARY ===\n");
fprintf("Agent mean cost     : %.3f\n", mean(agent_costs));
fprintf("Baseline mean cost  : %.3f\n", mean(baseline_costs));
fprintf("Agent mean comfort  : %.3f\n", mean(agent_comfort));
fprintf("Baseline comfort    : %.3f\n", mean(baseline_comfort));

%% -------------- PLOTS (last episode) ----------------
t_hours = agent_ts.steps/60;

figure('Name','Agent vs Baseline - Time Series','Position',[200 200 1200 700]);

subplot(3,2,1)
plot(t_hours, agent_ts.temps,'b', t_hours, baseline_ts.temps,'r')
yline(env.T_pref,'k--',"T_{pref}")
xlabel("Hours"), ylabel("Indoor Temp (°C)")
legend("Agent","Baseline"), title("Indoor Temperature")

subplot(3,2,2)
plot(t_hours, agent_ts.socs,'b', t_hours, baseline_ts.socs,'r')
xlabel("Hours"), ylabel("SOC")
legend("Agent","Baseline"), title("Battery SOC")

subplot(3,2,3)
plot(t_hours, agent_ts.grid,'b', t_hours, baseline_ts.grid,'r')
xlabel("Hours"), ylabel("Grid Power (kW)")
legend("Agent","Baseline"), title("Grid Power")

subplot(3,2,4)
plot(t_hours, agent_ts.prices,'b', t_hours, baseline_ts.prices,'r')
xlabel("Hours"), ylabel("Price ($/kWh)")
legend("Agent","Baseline"), title("Price Profile")

subplot(3,2,5)
plot(t_hours, cumsum(agent_ts.energy),'b', t_hours, cumsum(baseline_ts.energy),'r')
xlabel("Hours"), ylabel("Cumulative Cost ($)")
legend("Agent","Baseline"), title("Cumulative Energy Cost")

subplot(3,2,6)
stairs(t_hours, agent_ts.appl,'b'); hold on;
stairs(t_hours, baseline_ts.appl,'r')
xlabel("Hours"), ylabel("Appliance On (0/1)")
legend("Agent","Baseline"), title("Appliance Status")

%% ---------------- Helper: unwrapAction ----------------
function a = unwrapAction(a)
    if iscell(a), a=a{1}; end
    if isstruct(a) && isfield(a,'Data'), a=a.Data; end
    if isstring(a) || ischar(a), a=double(a); end
end

%% ---------------- BASELINE ROLLOUT (fixed) ----------------
function [total_cost, avg_comfort, ts] = baseline_rollout(env_in)

    dt_hours = env_in.Ts / 3600;    % <<< FIX HERE

    obs = reset(env_in);
    done = false;
    total_cost = 0;
    comfort_sum = 0;
    t = 0;

    temps=[]; outs=[]; socs=[]; prices=[]; gridP=[]; energy=[]; appl=[];

    while ~done
        hour = mod(t/60,24);

        if hour>=7 && hour<22
            HVAC=2; else HVAC=1;
        end
        if hour<6
            Batt=2;
        elseif hour>=16 && hour<20
            Batt=0;
        else
            Batt=1;
        end
        rem = double(obs(7));
        Appl = (rem<=0 && hour<6);

        % HVAC
        HVAC_p = env_in.HVAC_power_levels(HVAC+1);

        % battery
        if Batt==0, bcmd=-env_in.P_batt_max;
        elseif Batt==1, bcmd=0;
        else, bcmd=env_in.P_batt_max;
        end

        % appliance
        if Appl==1 && rem<=0
            app_now = env_in.appliance_power;
        elseif rem>0
            app_now = env_in.appliance_power;
        else
            app_now = 0;
        end

        % estimate grid power
        base=0.6;
        gp = base + HVAC_p + app_now;
        if bcmd>=0, gp=gp+bcmd; else gp=gp-max(0,-bcmd); end

        % step env
        a = int32(HVAC*6 + Batt*2 + Appl);
        [nextO, r, done] = step(env_in,a);

        temps(end+1)=obs(1);
        outs(end+1)=obs(2);
        socs(end+1)=obs(3);
        prices(end+1)=obs(6);
        gridP(end+1)=gp;
        energy(end+1)=gp * dt_hours * obs(6);
        appl(end+1)=app_now>0;

        total_cost = total_cost - r;
        comfort_sum = comfort_sum + abs(obs(1)-env_in.T_pref);

        obs = nextO;
        t = t+1;
    end

    avg_comfort = comfort_sum / max(1,t);

    ts.temps=temps;
    ts.outs=outs;
    ts.socs=socs;
    ts.prices=prices;
    ts.grid=gridP;
    ts.energy=energy;
    ts.appl=appl;
end
