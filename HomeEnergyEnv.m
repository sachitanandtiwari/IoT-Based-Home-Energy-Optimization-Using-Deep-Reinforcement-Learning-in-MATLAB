classdef HomeEnergyEnv < rl.env.MATLABEnvironment
    % HomeEnergyEnv: simple software-only home energy environment
    % Discrete action space: joint actions = HVAC(3) x Battery(3) x Appliance(2) = 18
    
    properties
        % Simulation parameters (tunable)
        Ts = 60;                % seconds (1-minute timestep)
        MinutesPerEpisode = 24*60; % 24 hours, 1-min steps
        T_out_base = 15;        % base outside temp (°C)
        T_out_amp = 8;          % amplitude of diurnal outside temp
        T_pref = 22;            % preferred indoor temp (°C)
        HVAC_power_levels = [0, 2, 5]; % kW for HVAC actions [OFF, ECO, COMFORT]
        P_batt_max = 3;         % kW max charge/discharge magnitude
        battery_capacity_kWh = 10; % kWh
        appliance_power = 1.5;  % kW when flexible appliance is running
        appliance_job_duration = 60; % minutes required for the flexible job
        price_offpeak = 0.08;   % $/kWh
        price_mid = 0.15;
        price_peak = 0.30;
        
        % Reward weights
        alpha_comfort = 6;    % comfort penalty multiplier
        beta_peak = 4;        % peak penalty multiplier
        gamma_batt = 0.05;    % battery usage penalty (to discourage cycling)
    end
    
    properties(Access = protected)
        % Internal states
        IndoorTemp       % current indoor temperature (°C)
        SOC              % battery state of charge (0..1)
        ApplianceRemain  % remaining minutes for flexible appliance (0 if none)
        timeStep         % current step number (1..MinutesPerEpisode)
        PriceProfile     % precomputed price vector for episode [$/kWh]
        OutsideProfile   % precomputed outside temp vector for episode
        LastGridPower    % for peak penalty shaping
    end
    
    properties(Access = public)
        % Observation and Action info are created in constructor (inherited)
    end
    
    methods
        function this = HomeEnergyEnv()
            % Observation: [IndoorTemp; OutsideTemp; SOC; time_of_day_sin; time_of_day_cos; price; appliance_remain]
            ObservationInfo = rlNumericSpec([7 1]);
            ObservationInfo.Name = 'observations';
            ObservationInfo.Description = 'IndoorTemp,OutsideTemp,SOC,sin_t,cos_t,price,appliance_remain';
            
            % Action: finite set 0..17 representing joint action:
            % encode index = HVAC_index*6 + Batt_index*2 + Appl_index
            ActList = int32(0:17);
            ActionInfo = rlFiniteSetSpec(ActList);
            ActionInfo.Name = 'actions';
            
            this = this@rl.env.MATLABEnvironment(ObservationInfo, ActionInfo);
            
            % Initialize/reset state
            reset(this)
        end
        
        function [Observation,Reward,IsDone,LoggedSignals] = step(this, Action)
            % Action is integer in 0..17 (rlFiniteSetSpec returns that)
            LoggedSignals = struct();
            IsDone = false;
            
            a = double(Action); % numeric
            
            % Decode joint action
            % HVAC_index in {0,1,2}, Batt_index in {0,1,2}, Appl_index in {0,1}
            HVAC_index = floor(a / 6);           % 0..2
            rem = mod(a,6);
            Batt_index = floor(rem / 2);         % 0..2
            Appl_index = mod(rem,2);            % 0..1
            
            % Map to actual control signals
            HVAC_power = this.HVAC_power_levels(HVAC_index+1); % kW
            % Battery command mapping: 0->discharge,1->hold,2->charge ; map to power (kW)
            if Batt_index == 0
                batt_cmd_kW = -this.P_batt_max; % discharge (negative means supplying house)
            elseif Batt_index == 1
                batt_cmd_kW = 0;
            else
                batt_cmd_kW = +this.P_batt_max; % charge
            end
            % Appliance action: 1 => start if not already running; 0 => do nothing
            if Appl_index == 1 && this.ApplianceRemain <= 0
                this.ApplianceRemain = this.appliance_job_duration;
            end
            
            % Environment dynamics (1-minute Euler)
            dt_hours = this.Ts/3600; % hours step (1/60)
            idx = this.timeStep;
            Tout = this.OutsideProfile(idx);
            % simple RC thermal: T_next = T + dt/C*( - (T - Tout)/R + HVAC_effect )
            % choose C=1, R=3 for simple behavior -> time constants in tens of minutes
            C = 1.0; R = 3.0;
            HVAC_cop = 1.0; % convert HVAC power (kW) to heating/cooling effect on temperature
            % Assume HVAC power positive means cooling/heating towards T_pref.
            % Simpler: HVAC acts to drive temperature toward T_pref proportionally to power.
            % deltaT_from_HVAC = (T_pref - T)/abs(T_pref - T + 1e-6) * (HVAC_power * dt_hours)
            % We'll use a linear effect:
            T = this.IndoorTemp;
            heat_loss = -(T - Tout)/R;
            hvac_effect = (this.T_pref - T) * (HVAC_power * 0.2); % tuning constant
            dT = (heat_loss + hvac_effect)/C * (this.Ts/60); % per minute scaling
            T_next = T + dT;
            
            % Battery SOC update: batt_cmd_kW (positive -> charge), battery_capacity_kWh
            soc = this.SOC;
            energy_change_kWh = batt_cmd_kW * dt_hours; % can be negative
            soc_next = soc + energy_change_kWh / this.battery_capacity_kWh;
            % clamp SOC
            soc_next = max(0, min(1, soc_next));
            
            % Appliance power
            if this.ApplianceRemain > 0
                appliance_power_now = this.appliance_power; % kW
                this.ApplianceRemain = max(0, this.ApplianceRemain - 1); % one minute passes
            else
                appliance_power_now = 0;
            end
            
            % Building base load (non-controllable) and HVAC load:
            base_load = 0.6; % kW constant background (fridge etc.)
            building_power = base_load + HVAC_power; % HVAC_power is in kW
            % Total grid power draw: building + appliance - battery discharge (if battery supplies)
            % Convention: positive grid power = drawn from grid
            % If batt_cmd_kW < 0 (discharge), battery provides power, reducing grid draw.
            grid_power = building_power + appliance_power_now - max(0,-batt_cmd_kW);
            % If batt_cmd_kW >0 (charging), battery draws additional power
            if batt_cmd_kW > 0
                grid_power = grid_power + batt_cmd_kW;
            end
            
            % Energy cost for this minute (kWh * $/kWh)
            price = this.PriceProfile(idx);
            energy_kWh = grid_power * dt_hours; % could be negative (net export), charge user negative cost if so
            energy_cost = energy_kWh * price;
            
            % Comfort penalty: deviation beyond a small deadband
            temp_dev = abs(T_next - this.T_pref);
            comfort_pen = max(0, temp_dev - 0.5); % tolerance 0.5 deg
            comfort_cost = this.alpha_comfort * comfort_pen;
            
            % Peak penalty (large penalty for large instantaneous grid draw relative to avg)
            peak_cost = this.beta_peak * max(0, grid_power - 6); % penalize >6 kW draws
            
            % Battery usage penalty to discourage excessive cycling
            batt_cycle_cost = this.gamma_batt * abs(soc_next - soc) * this.battery_capacity_kWh;
            
            % Reward: negative of costs (we maximize reward)
            Reward = - (energy_cost + comfort_cost + peak_cost + batt_cycle_cost);
            
            % Update internal state
            this.IndoorTemp = T_next;
            this.SOC = soc_next;
            this.LastGridPower = grid_power;
            this.timeStep = this.timeStep + 1;
            
            % Observation
            Observation = this.getObservation();
            
            % Check done
            if this.timeStep > this.MinutesPerEpisode
                IsDone = true;
            else
                IsDone = false;
            end
        end
        
        function InitialObservation = reset(this)
            % Randomize start conditions slightly
            this.timeStep = 1;
            this.IndoorTemp = this.T_pref + (rand-0.5)*1.0; % near preferred temp
            this.SOC = 0.5 + (rand-0.5)*0.1;
            this.ApplianceRemain = 0;
            this.LastGridPower = 0;
            
            % Build daily profiles (sinusoid for outside temp, simple TOU price tiers)
            t = (0:this.MinutesPerEpisode-1)/60; % hours
            this.OutsideProfile = this.T_out_base + this.T_out_amp * sin( (2*pi/24) * (t - 6) ); % cold at night, warm midday
            % Price profile: low at night (0-6), mid (6-16), peak (16-20), mid (20-24)
            this.PriceProfile = zeros(1,this.MinutesPerEpisode);
            for i=1:length(t)
                hour = mod(t(i),24);
                if hour >=0 && hour < 6
                    this.PriceProfile(i) = this.price_offpeak;
                elseif hour >=6 && hour < 16
                    this.PriceProfile(i) = this.price_mid;
                elseif hour >=16 && hour < 20
                    this.PriceProfile(i) = this.price_peak;
                else
                    this.PriceProfile(i) = this.price_mid;
                end
            end
            
            InitialObservation = this.getObservation();
        end
    end
    
    methods (Access = protected)
        function obs = getObservation(this)
            t = (this.timeStep-1)/60; % hours into day
            sin_t = sin(2*pi*t/24);
            cos_t = cos(2*pi*t/24);
            idx = min(max(1,this.timeStep), this.MinutesPerEpisode);
            price = this.PriceProfile(idx);
            obs = [ this.IndoorTemp;
                    this.OutsideProfile(idx);
                    this.SOC;
                    sin_t;
                    cos_t;
                    price;
                    double(this.ApplianceRemain) ];
        end
    end
end
