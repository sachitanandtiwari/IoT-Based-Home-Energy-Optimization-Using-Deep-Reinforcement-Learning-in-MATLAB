function [total_cost, avg_comfort] = my_baseline_policy(env)
% Clean baseline policy (no hidden characters)

    obs = reset(env);
    done = false;
    total_cost = 0;
    comfort_sum = 0;
    stepCount = 0;

    while ~done
        hour = mod(stepCount/60, 24);

        % HVAC
        if hour >= 7 && hour < 22
            HVAC_index = 2;
        else
            HVAC_index = 1;
        end

        % Battery
        if hour >= 0 && hour < 6
            Batt_index = 2;
        elseif hour >= 16 && hour < 20
            Batt_index = 0;
        else
            Batt_index = 1;
        end

        % Appliance: obs(7)
        appliance_remain = double(obs(7));
        if appliance_remain <= 0 && hour >= 0 && hour < 6
            Appl_index = 1;
        else
            Appl_index = 0;
        end

        % Encode action
        a = int32(HVAC_index*6 + Batt_index*2 + Appl_index);

        [nextObs, reward, done, ~] = step(env, a);

        total_cost = total_cost - reward;
        comfort_sum = comfort_sum + abs(obs(1) - env.T_pref);

        obs = nextObs;
        stepCount = stepCount + 1;
    end

    avg_comfort = comfort_sum / max(1, stepCount);
end
