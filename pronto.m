function [x, u, output] = pronto2(x0, u0, x_des, u_des, dyn_params, cost_params, function_handles, opts)

addpath('helper')
definitions;

plot_intermediates = opts.diagnostics;      % Sub plots after each iteration
if plot_intermediates
  figure(1)
  xxx = subplot(2,3,1); yyy = subplot(2,3,2); zzz = subplot(2,3,3);
  set(gcf, 'Position', get(0, 'Screensize'));
end

%% Algorithm parameters

% Default configuration
print_diagnostics  = true;  % print diagnostics at each iteration
[n, N] = size(x0);          % n: state dimension, N: number of stages
m      = size(u0, 1);       % number of inputs

AbsTol = 1e-4;              % absolute tolerance convergence criterion
numFirstOrderIters = 3;     % set how many first order step to take

maxConstraintIter = 25;     % maximum number of constraint iterations
maxPRONTOIter = 100;           % maximum number of iterations
% line search params
maxStepLenghtIter = 25;     %
alpha = 0.4;                % armijo's rule parameter
beta = 0.7;                 % backtracking constant
% approx barrier params
delta0 = 0.5;               % initial delta
delta_step = 0.1;
% continuation parameters 
epsilon0 = 1;               % initial epsilon
epsilon_min = 1e-5;
epsilon_step = 0.1;

% AS 26 Aug 2017 - these paramaters not fully understood/checked yet...
scale_P_N_prj = 1;          % scaling factor for terminal gain projection op.
redesign_prj = false;       % redesign projection operator after each line search iter
enable_prj = true;          % Enable projection operator (disable -> SNM)
project_init_trajectory = false;
discardSecondOrderDynamics = false;

% Read non-default parameters from opts struct and overwrite default values
if isfield(opts,'t_f'),           t_f           = opts.t_f;           end
if isfield(opts,'AbsTol'),        AbsTol        = opts.AbsTol;        end
if isfield(opts,'MaxIter'),       maxPRONTOIter = opts.MaxIter;       end
if isfield(opts,'print_diagnotics'), print_diagnostics = strcmp(opts.print_diagnostics,true); end
if isfield(opts,'Q_prj'),         Q_prj         = opts.Q_prj;         end
if isfield(opts,'R_prj'),         R_prj         = opts.R_prj;         end
if isfield(opts,'alpha'),         alpha         = opts.alpha;         end
if isfield(opts,'beta'),          beta          = opts.beta;          end
if isfield(opts,'k_max'),         k_max         = opts.k_max;         end
if isfield(opts,'delta0'),        delta0        = opts.delta0;        end
if isfield(opts,'epsilon0'),      epsilon0      = opts.epsilon0;      end
if isfield(opts,'delta_step'),    delta_step    = opts.delta_step;    end
if isfield(opts,'epsilon_step'),  epsilon_step  = opts.epsilon_step;  end
if isfield(opts,'epsilon_min'),   epsilon_min   = opts.epsilon_min;   end
if isfield(opts,'scale_P_N_prj'), scale_P_N_prj = opts.scale_P_N_prj; end
if isfield(opts,'redesign_prj'),  redesign_prj  = opts.redesign_prj;  end
if isfield(opts,'enable_prj'),    enable_prj    = opts.enable_prj;    end
if isfield(opts,'project_init_trajectory'), project_init_trajectory  = opts.project_init_trajectory; end
if isfield(opts,'discard_second_order_dynamics'), discardSecondOrderDynamics = opts.discard_second_order_dynamics; end

% Read struct of function handles
h_cost   = function_handles.cost;
h_cost12 = function_handles.cost12;
h_dyn    = function_handles.dynamics;
h_dyn12  = function_handles.dynamics12;
if isfield(function_handles,'constraints')
    h_con    = function_handles.constraints;
    h_con12  = function_handles.constraints12;
else
    h_con = [];
    h_con12 = [];
end

% Compute step size
hh = t_f/(N-1);

%% Algorithm

%[x, u] = projectionOperator(x0, u0, Q_prj, R_prj, hh, h_dyn, h_dyn12);

if project_init_trajectory 
    % Determine intitial trajectory by trajecting (??) initial curve 
    % Choose terminal gain
    [F_x, F_u] = dynamics12(h_dyn12, x0, u0, dyn_params, hh);
    try
        [~, P_prj(:,:,N)] = dlqr(F_x(:,:,N), F_u(:,:,N), Q_prj, R_prj);
        P_prj(:,:,N) = scale_P_N_prj*P_prj(:,:,N);
    catch
        error('something wrong with designing the regulator');
        disp('Unable to find suitable P');
        P_prj(:,:,N) = 2000*eye(n); % hack ...
    end
    K_prj = zeros(m,n,N-1);
    
    % Design projection operator gain (AS Jun 15, 2017: to be improved...)
    for kk=N-1:-1:1
        f_x = F_x(:,:,kk); f_u = F_u(:,:,kk);
        P_prj(:,:,kk) = f_x'*P_prj(:,:,kk+1)*f_x + Q_prj - (f_x'*P_prj(:,:,kk+1)*f_u)*inv(f_u'*P_prj(:,:,kk+1)*f_u + R_prj)*(f_u'*P_prj(:,:,kk+1)*f_x);
        K_prj(:,:,kk) = inv(f_u'*P_prj(:,:,kk+1)*f_u + R_prj)*(f_u'*P_prj(:,:,kk+1)*f_x);
        
        % P_prj(:,:,kk) = Q_prj + f_x'*(P_prj(:,:,kk+1) - P_prj(:,:,kk+1)'*f_u*(R_prj + f_u'*P_prj(:,:,kk+1)*f_u)^(-1)*f_u'*P_prj(:,:,kk+1))*f_x;
        % K_prj(:,:,kk) = (R_prj + f_u'*P_prj(:,:,kk)*f_u)^(-1)*f_u'*P_prj(:,:,kk)*f_x;
    end
    
    % Compute projected trajectory
    x = zeros(n,N);  x(:,1) = x0(:,1);
    u = zeros(m,N-1);
    for kk=1:N-1
        u(:,kk)   = u0(:,kk) +  K_prj(:,:,kk)*(x0(:,kk) - x(:,kk));
        x(:,kk+1) = dynamics(h_dyn, x(:,kk), u(:,kk), dyn_params, hh);
    end
else
    % Determine trajectory using (x_0, u_0:N-1)
    % AS Jun 16, 2017 I think this open loop integration is very dangerous   
    %                 ... as well as not necessary as x0 should be already
    %                     a trajectory
    x = zeros(n,N); x(:,1) = x0(:,1);
    for kk = 1:N-1
      x(:,kk+1) = dynamics(h_dyn, x(:,kk), u0(:,kk), dyn_params, hh);
    end
    u = u0;
end


if 1 % compare initial trajector with its projection
  t_vec = hh*[0:N-1];
  figure(10000000)
  subplot(211)
    plot(t_vec, x0, t_vec, x)
    grid on
  subplot(212)
    plot(t_vec(1:end-1), u0, t_vec(1:end-1), u)
    grid on
  pause
end


figure(1000)
  hold on, grid on
  plot3(x(1,:),x(3,:),x(5,:),'linewidth',2);

output{1}.x = x;
output{1}.u = u;
output{1}.z = zeros(size(x)); % initialize z to zero
output{1}.v = zeros(size(u)); % initialize v to zero
output{1}.cost    = 0;
output{1}.descent = 0;
output{1}.gamma   = 0;
output{1}.delta   = delta0;
output{1}.epsilon = epsilon0;

delta   = delta0;
epsilon = epsilon0;

fprintf('Initial cost: %f\n', costfun(h_cost, h_con, x, u, x_des, u_des, cost_params, delta, epsilon));
print_diagnostics_header(print_diagnostics);

% preallocation
%f_x = zeros(n,n); f_u = zeros(n,m); f_xx = zeros(n,n,n); f_ux = zeros(n,m,n); f_uu = zeros(m,m,n);
%l_x = zeros(n,1); l_u = zeros(m,1); l_xx = zeros(n,n); l_ux = zeros(n,m); l_uu = zeros(m,m);

% **** Top-level iterations for constraint values *************************
%
% Only do top-level iterations when solving constrained problems, i.e., 
% when a constraint function h_con is supplied

%if isfield(dyn_params, 'update_k_v')
%    update_k_v =  dyn_params.update_k_v;
%else
%    update_k_v = false;
%end


% AS 26 Aug 2017
% This is the interior point method outer loop that runs outside dPRONTO
% 

% ORIGINAL CODE by Remon ( I simplified it, also removed update_k_v...)
% for mm= 1: 1+(maxConstraintIter-1)*(isa(h_con, 'function_handle') || update_k_v)
  
if ~isa(h_con, 'function_handle') % if there are no constraints...
  maxConstraintIter = 1; % ... then there is no reason for updating epsilon
                         % and having an outer loop around dPRONTO
end


for mm = 1: maxConstraintIter % this loop updates epsilon (continuation)
    
  % Flag becomes true the first time delta is set to delta_min; is used so  
  % that the first time delta is reset, the inner loop does at least one
  % iteration with this new value for delta. 
  isDeltaSetToMin = false;
    
  
%   figure(10000)
%   pause
  
  
  %%%%%%% dPRONTO iterations (BEGIN) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
  for ii = 1:maxPRONTOIter 
    
    % AS 26 Aug 2017 - is this DeltaSetToMin necessary??
        
    % Obtain linearizations
    c = costfun(h_cost, h_con, x, u, x_des, u_des, cost_params, delta, epsilon);
        
    % AS Jun 16, 2017 
    % It is a pity: the implementation is hybrid and should be improved...
    % Sometimes the function return a vector
    % sometimds the function return a scalar (as in the case of costfun)
    %
    % Add a figure to how the integrant related to cost is
        
    if 1 % plot_intermediates    
      figure(200), clf
        subplot(311)
          plot(running_cost(x,u, x_des, u_des, cost_params))  % a HACK ... missing function in Remon implementation
          title('l(x,u)')
          grid on, zoom on                 
        subplot(312)        
          if isa(h_con, 'function_handle')
            plot(epsilon*betadelta(delta,h_con(x,u)))
          end
        title('\epsilon \beta_\delta (c(x,u))')
          grid on, zoom on
        subplot(313)
          if isa(h_con, 'function_handle')
            plot(h_con(x,u))
          end
          title('c(x,u)')
          grid on, zoom on         
      %figureEND
        
      % some debug code ... to verify the constraints 
      % and log barrier computation (some entries become Inf for delta
      % equal to zero... 
      %
      % delta   
      % h_con(x,u)
      % epsilon*betadelta(delta,h_con(x,u))
      % pause
     
    end
    
    % sensitivity and Hessian of the dynamics and the cost about (x,u)
    [F_x, F_u, F_xx, F_ux, F_uu] = dynamics12(h_dyn12, x, u, dyn_params, hh);
    [L_x, L_u, L_xx, L_ux, L_uu] = cost12(h_cost12, h_con, h_con12, x, u, x_des, u_des, cost_params, delta, epsilon);
        
        
    % AS Jun 16, 2017 Apparently, Remon is redesigning the
    %                 proj operator gain at every iteration
        
    P_prj = zeros(n,n,N);
    try
      [~, P_prj(:,:,N)] = dlqr(F_x(:,:,N), F_u(:,:,N), Q_prj, R_prj);
      P_prj(:,:,N) = scale_P_N_prj*P_prj(:,:,N);
    catch
      disp('Unable to find suitable P');
      F_x(:,:,N), F_u(:,:,N), Q_prj, R_prj
      pause
      P_prj(:,:,N) = 2000*eye(n); % just a hack
    end
    K_prj = zeros(m,n,N-1);
        
    V_xx(:,:,N) = L_xx(:,:,N);
    V_x(:,N)    = L_x(:,N); % - P_prj(:,:,N)*x_des(:,N);
    q(:,N)      = L_x(:,N); % - P_prj(:,:,N)*x_des(:,N);
    theta       = zeros(N,1);
        
    E = zeros(m,n,N-1);
    F = zeros(m,N-1);
        
    isPD_count = 0; % Keeps track of the percentage (%) of PD time steps
        
    % ---- Backward sweep ---------------------------------------------
        
    % The input is of a recursive form, and the inputs gains E_k and
    % F_k are computed backward in time. For each time step, we compute
    % the cost coefficients V_xx, V_x, theta; the input gains E_k, F_k;
    % and q_k, which servers as Lagrange multiplier / adjoint.
     
        
    % these plots shows the first and secodn derivatives 
    % of the running cost (actually, also terminal ... strange choice 
    % from Remon). These are the real game changers, as the dynamics
    % for Bram's project is just a second order integrator and 
    % it' rather trivial
        
    if plot_intermediates
      figure(300), clf
        subplot(211), hold on 
          for rrr = 1:n
            plot(reshape(L_x(rrr,:),1,N))
          end
            title('D1l')
        subplot(212), hold on
          for rrr = 1:m
            plot(reshape(L_u(rrr,:),1,N-1))
          end
        title('D2l')
      %figureEND
      figure(400), clf
        subplot(311), hold on
          for rrr = 1:n
            for ccc = 1:n
              plot(reshape(L_xx(rrr,ccc,:),1,N))
            end
          end
        title('DD1l')
        subplot(312), hold on
          for rrr = 1:m
            for ccc = 1:m
              plot(reshape(L_uu(rrr,ccc,:),1,N-1))
            end
          end
        title('DD2l')
        subplot(313), hold on
          for rrr = 1:m
            for ccc = 1:n
              plot(reshape(L_ux(rrr,ccc,:),1,N-1))
            end
          end
        title('DD12l')
      %figureEND
    end
        
    for kk = N-1:-1:1
      % Store linearizations in a more convenient form. Note that
      % linearizations are computed all at once to reduce the number
      % of Matlab function calls (uses more memory though)
      f_x = F_x(:,:,kk); 
      f_u = F_u(:,:,kk); 
            
      f_xx = F_xx(:,:,:,kk); 
      f_ux = F_ux(:,:,:,kk); 
      f_uu = F_uu(:,:,:,kk);
            
      l_x = L_x(:,kk); 
      l_u = L_u(:,kk); 
            
      % AS Jun 16, 2017 - Force 1st order descent for the first
      % iterations
      if (ii < numFirstOrderIters)
        L_xx(:,:,kk) = hh * hh * 1000 * eye(n);
        L_ux(:,:,kk) = zeros(size(L_ux(:,:,kk)));
        L_uu(:,:,kk) = hh * hh * 1000 * eye(m);
      end
      
      l_xx = L_xx(:,:,kk);   
      l_ux = L_ux(:,:,kk); 
      l_uu = L_uu(:,:,kk); 
            
      % Compute projection operator gain
      if enable_prj
        P_prj(:,:,kk) = f_x'*P_prj(:,:,kk+1)*f_x + Q_prj - (f_x'*P_prj(:,:,kk+1)*f_u)/(f_u'*P_prj(:,:,kk+1)*f_u + R_prj)*(f_u'*P_prj(:,:,kk+1)*f_x);
        K_prj(:,:,kk) = (f_u'*P_prj(:,:,kk+1)*f_u + R_prj)\f_u'*P_prj(:,:,kk+1)*f_x;
      end
      
      % First compute tensor products and store in q_**
      q(:,kk) = l_x - K_prj(:,:,kk)'*l_u + (f_x - f_u*K_prj(:,:,kk))'*q(:,kk+1);      
      
      % AS 26 Aug 2017 - Calling these matrices q should be banned!
      % Need to change these into something like qD2f or similar....
      
      q_xx = zeros(n,n);
      q_ux = zeros(m,n);      
      q_uu = zeros(m,m);
      for jj=1:n
        q_xx = q_xx + q(jj,kk+1)'*f_xx(:,:,jj);
        q_ux = q_ux + q(jj,kk+1)'*f_ux(:,:,jj);        
        q_uu = q_uu + q(jj,kk+1)'*f_uu(:,:,jj);
      end
      
      % AS 26 Aug 2017 - Trying to fix what I think is a bug in the
      % computation of desc2 (later on). Create a matrix W combining
      % l and qD2f. Definitely not computationally efficient!!
      
      W_xx(:,:,kk) = l_xx + q_xx;
      W_ux(:,:,kk) = l_ux + q_ux;
      W_uu(:,:,kk) = l_uu + q_uu;
      
     
      % For nonlinear dynamics, it might happen that no descent
      % direction is found. We therefore check positive definiteness
      % of the Hessian and of the quadratic component of the
      % cost-to-go. If one of these is not positive definite, revert
      % to the 1.5-order search directiom i.e., take only a linear
      % approximation of the dynamics.
            
      % AS Jun 16, 2017 Here Remon is doing something unclear
      %                 Positive definiteness of the quadratic form
      %                 on the linear subspace identified is
      %                 guaranteed, in continuous-time, by the
      %                 existence of the solution of the differential
      %                 Riccati equation.            %
      %                 The check Remon has implemented seems an
      %                 heuristic, not fully motivated theoretically 
      %                 Q does NOT need to be positive definite 
      %                 at each time step ... this is restrictive !!            
            
      % AS 26 Aug 2017 - I have reordered terms to keep
      % l_xx + q_xx together (actually, q is a bad name and it should be 
      % changed as it represents both q and D^2 f ... ) 
      
      Q_xx = l_xx + q_xx + f_x'*V_xx(:,:,kk+1)*f_x;
      Q_ux = l_ux + q_ux + f_u'*V_xx(:,:,kk+1)*f_x;
      Q_uu = l_uu + q_uu + f_u'*V_xx(:,:,kk+1)*f_u;
      V_xx(:,:,kk) = Q_xx - Q_ux'/Q_uu*Q_ux;
            
%     V_xx(:,:,kk) % AS Jun 16, 2016 This should be the 
%                  % the solution of the Riccati equation
            
            
      [~, isNotPdQuu] = chol(Q_uu + 1e-6*eye(m));
      [~, isNotPdVxx] = chol(V_xx(:,:,kk) + 1e-6*eye(n));
            
      if isNotPdQuu || isNotPdVxx || discardSecondOrderDynamics
        isConvex = false;
        disp(' Q is not positive definite! ')
      else
        isConvex = true;       %TODO: set to true
      end
            
      % Compute cost coefficients (some matrices are computed twice,
      % but I left it like this for readability).
            
      isPD_count = isPD_count + (isConvex == true)/(N-1);
            
      % AS Jun 16, 2017 Remon implemented the "1.5 descent direction"
      %                 I am here deliberately setting isConvex to 
      %                 zero to force this to be used 
            
      % AS Jun 16, 2017
      % force 1.5 order descent formulas for small ii
      if (ii < numFirstOrderIters)
        isConvex = 0; 
      else
        isConvex = 1;  
      end
      
      Q_x  = l_x  + f_x'*V_x(:,kk+1);
      Q_u  = l_u  + f_u'*V_x(:,kk+1);
      Q_xx = l_xx + (isConvex)*q_xx + f_x'*V_xx(:,:,kk+1)*f_x;
      Q_ux = l_ux + (isConvex)*q_ux + f_u'*V_xx(:,:,kk+1)*f_x;
      Q_uu = l_uu + (isConvex)*q_uu + f_u'*V_xx(:,:,kk+1)*f_u;
      
      % AS 26 Aug 2017 - Just added to mimick what Remon did... 
      W_xx(:,:,kk) = l_xx + (isConvex)*q_xx;
      W_ux(:,:,kk) = l_ux + (isConvex)*q_ux;
      W_uu(:,:,kk) = l_uu + (isConvex)*q_uu;
        
      
      % Compute gain and cost matrices
      E(:,:,kk) = -Q_uu\Q_ux;
      F(:,kk)   = -Q_uu\Q_u;
            
      V_xx(:,:,kk) = Q_xx - Q_ux'/Q_uu*Q_ux;
      V_x(:,kk)    = Q_x  - Q_ux'/Q_uu*Q_u;
            
      theta(kk)    = theta(kk+1) - 0.5*Q_u'/Q_uu*Q_u;
            
      % Make V_xx symmetric (it loses symmetry due to round-off errors)
      
      % AS 26 Aug 2017 - that is wht in the original PRONTO implementation
      % the Riccati equation was run just on the upper triangular part
      
      V_xx(:,:,kk) =(V_xx(:,:,kk) + V_xx(:,:,kk).')/2;
    end
    % -----------------------------------------------------------------
        
        
    % AS Jun 16, 2017 Debug code to see the Riccati equation solution
    figure(100), clf
      for rrr = 1:n
        for ccc = 1:n
          hold on, plot(reshape(V_xx(rrr,ccc,:),1,N))
        end              
      end
    title('P_k Riccati difference equation (RDE)')  
    grid on, zoom on
    
    figure(1)
    
    
    % Compute associated zeta = (z,v) discent direction
    
    z       = zeros(n, N);        % Implies initial condition z(:,1) = 0;
    v       = zeros(m, N-1);      % recall that in disc time with ZOH,
                                    % input dimension is smaller than state
            
    desc1 = 0; % FIRST derivative of the cost in the descent direction
               % desc1 should always match with the first derivative
               % of the nonlinear cost
                 
    desc2 = 0; % SECOND derivative of the cost in the descent direction
               % desc2 matches the second derivative of the nonlinear cost
               % only when 2nd order derivs are fully used
               
    % Compute linear trajectory (solution to LQ problem)
            
    for kk=1:N-1
      % AS 26 Aug 2017 - E is the optimal feedback, F is the feedforward
      %                  They are the solution of the Riccati equation
      %                  and the feedforward equation due to the linear
      %                  term in the LT problem
        
      v(:,kk)   = E(:,:,kk)*z(:,kk) + F(:,kk); % optimal feedback
      z(:,kk+1) = F_x(:,:,kk)*z(:,kk) + F_u(:,:,kk)*v(:,kk); % z^+ = Az + Bv

      desc1 = desc1 + L_x(:,kk)'*z(:,kk) + L_u(:,kk)'*v(:,kk);
                
      % AS Jun 17, 2017 I must say I am PUZZLED not to see the
      % the presence of the multiplier q ...
      % *** NEED TO CHECK THE CORRECTNESS OF THESE FORMULAS ***
      % Alfter all, this should be the matrix W in the standard
      % implementation of PRONTO (Hessian l + q_i Hessian f_i)
       
      desc2 = desc2 + 1/2 * z(:,kk)'*L_xx(:,:,kk)*z(:,kk) ...
                      +       v(:,kk)'*L_ux(:,:,kk)*z(:,kk) ...
                      + 1/2 * v(:,kk)'*L_uu(:,:,kk)*v(:,kk);
                    
      % AS 26 Aug 2017 - Possible fix ? No difference for linear dynamics
      % as in that case D2f are all zeros
      
      %         desc2 = desc2 + 1/2 * z(:,kk)'*W_xx(:,:,kk)*z(:,kk) ...
      %                       +       v(:,kk)'*W_ux(:,:,kk)*z(:,kk) ...
      %                       + 1/2 * v(:,kk)'*W_uu(:,:,kk)*v(:,kk);
      
                    
    end
            
    % add contribution of the terminal cost
    %
    % AS 26 Aug 2017 - Once again, this programming style should be
    % changed... storing even the derivative of the terminal cost in L_x
    % is very confusiong and prone to generate bugs/misinterpretations
    %
    desc1 = desc1 + L_x(:,N)'*z(:,N);
    desc2 = desc2 + 1/2 * z(:,N)'*L_xx(:,:,N)*z(:,N);
    
    epsilon
    delta
    
    desc1
    desc2
            
    
  
    % AS 30 Aug 2017 - Visualize the descent direction on the 3D figure
    
    figure(1000)
      hold on, grid on
      plot3(x(1,:)+z(1,:),x(3,:)+z(3,:),x(5,:)+z(5,:),'r--');
    %figure end    
    
    
  
    
    % --- Forward sweep -----------------------------------------------
    % PRONTO essentially has two forward sweeps. First a trajectory of
    % the linearized dynamics is computed. This trajectory serves as
    % the descent direction in the second forward sweep. The second
    % sweep computes a projection of:
    % PROJ[(nominal trajectory) + (step length)*(descent direction)]
    % and uses Armijo's rule to determine a suitable step length.
    
    
    gamma = 1; % this is the step length (when = 1, it means full step)
     
    
    % This is the backtracking line search using Armijo's rule
    for jj = 1:maxStepLenghtIter 
 
            
      % AS Jun 16, 2017. The following was nonsense. 
      %                  The projection operator should not be
      %                  redesigned about xi + gamma*zeta !!!
      %%%%%%%%%       
      %        
      %        if enable_prj && redesign_prj
      %            % Redesign projection operator gain for xi+gamma*zeta
      %            try
      %                [~, P_prj(:,:,N)] = dlqr(F_x(:,:,N), F_u(:,:,N), Q_prj, R_prj); 
      %                P_prj(:,:,N) = scale_P_N_prj*P_prj(:,:,N);
      %            catch
      %                disp('Unable to find suitable P_N');
      %                P_prj(:,:,N) = 2000*eye(n);
      %            end
      %            [F_x_gamma, F_u_gamma] = dynamics12(h_dyn12, x+gamma*z, u+gamma*v, dyn_params, hh);
      %            for kk=N-1:-1:1
      %                f_x = F_x_gamma(:,:,kk); f_u = F_u_gamma(:,:,kk);
      %                P_prj(:,:,kk) = f_x'*P_prj(:,:,kk+1)*f_x + Q_prj - (f_x'*P_prj(:,:,kk+1)*f_u)/(f_u'*P_prj(:,:,kk+1)*f_u + R_prj)*(f_u'*P_prj(:,:,kk+1)*f_x);
      %                K_prj(:,:,kk) = (f_u'*P_prj(:,:,kk+1)*f_u + R_prj)\f_u'*P_prj(:,:,kk+1)*f_x;
      %            end
      %        end
            
    
      % Compute projected trajectory xi_gamma = P (xi + gamma*zeta)
            
      x_gamma = zeros(n,N); x_gamma(:,1) = x(:,1);
      u_gamma = zeros(m,N-1);
       
      % AS 26 Aug 2017 - running the dynamics in closed loop should be executed as a
      % separate routine (in PRONTO, there are S-functions run in a
      % simulink block)
      for kk=1:N-1
        u_gamma(:,kk) = (u(:,kk) + gamma*v(:,kk)) +  K_prj(:,:,kk)*((x(:,kk) + gamma*z(:,kk)) - x_gamma(:,kk));
        x_gamma(:,kk+1) = dynamics(h_dyn, x_gamma(:,kk), u_gamma(:,kk), dyn_params, hh);   
      end
      % compute cost associated to xi_gamma = P (xi + gamma*zeta)
      c_gamma = costfun(h_cost, h_con, x_gamma, u_gamma, x_des, u_des, cost_params, delta, epsilon);
            
 
      % AS 30 Aug 2017 New debug plot (x, u, z, v, and cost along (z,v))
      if 1
        plot_colors = 'kbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmy';
        t = linspace(0, t_f, N);
        
        figure(500)
        subplot(231),cla, hold on
          for bb=1:n % state 
            plot(t, x(bb,:), plot_colors(bb));
          end
        grid on, zoom on
        subplot(232), cla,  hold on
          for bb=1:m % input
            plot(t(1:end-1), u(bb,:), plot_colors(bb));
          end
        grid on, zoom on  
       
        subplot(234), cla, hold on % discent direction state
          for bb=1:n
            plot(t, z(bb,:), plot_colors(bb));
          end
        grid on, zoom on
        subplot(235), cla, hold on % discent directin input
          for bb=1:m
            plot(t(1:end-1), v(bb,:), plot_colors(bb));
          end
        grid on, zoom on
    
        
        % Compute h ( P( xi + gamma zeta ) ) for different gammas 
        gamma_vec= 0:0.005:1;
        xx_prj = zeros(n,N); xx_prj(:,1) = x(:,1);
        uu_prj = zeros(m,N-1);
        desc1_temp = []; desc2_temp = 0;
        for bb = 1:length(gamma_vec)
          for cc=1:N-1
            uu_prj(:,cc) = (u(:,cc) + gamma_vec(bb)*v(:,cc)) + K_prj(:,:,cc)*((x(:,cc) + gamma_vec(bb)*z(:,cc)) - xx_prj(:,cc));
            xx_prj(:,cc+1) = dynamics(h_dyn, xx_prj(:,cc), uu_prj(:,cc), dyn_params, hh);
          end
          costs_prj(bb) = costfun(h_cost, h_con, xx_prj, uu_prj, x_des, u_des, cost_params, delta, epsilon);
        end
        
        % Plot descent in cost
        if jj == 1
          subplot(2,3,6), cla, grid on, hold on
            plot(gamma_vec, c + gamma_vec*desc1 + gamma_vec.^2*desc2, 'k--');
            plot(gamma_vec, costs_prj, 'k');
            plot(gamma,c_gamma,'o',0,c,'*','MarkerFaceColor',[0 0 0],'MarkerSize',5)
            h = fill([0, 0, 1, 1], [c, c + desc1, c + desc1, c + alpha*desc1], 'b');
            set(h,'FaceAlpha',0.1);
            set(h,'EdgeColor','None');
            xlabel('Step length [-]'), ylabel('Cost [-]'), hold off
            
          title(sprintf('Change in cost  (iteration %i,%i)', ii, jj))
        else
          subplot(2,3,6), hold on
            plot(gamma, c_gamma, 'o','MarkerFaceColor',[0 0 0],'MarkerSize',5)
            hold off
          title(sprintf('Change in cost  (iteration %i,%i)', ii, jj))
        end
        
          % AS 26 Aug 2017 - Now that I think about, the missing q terms
          % in the computations of the running cost of the LQ problem could
          % explain why the gradient of the cost desc1 sometimes becomes positive
          % (nonsense!) and consequently the following alert becomes active
          if desc1 > 0
            error('something wrong - positive descent!')
          end
        
        pause
      end
      
      
      
      % Plot diagnostics --------------------------------------------
      if plot_intermediates
        plot_colors = 'kbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmy';
        
        % Plot states (old, new, projection)
        t = linspace(0, t_f, N);
        subplot(2,3,1), cla, grid on, hold on
        for bb=1:n
          plot(t, x(bb,:),                 [plot_colors(bb) '-.']);
          plot(t, x(bb,:) + gamma*z(bb,:), plot_colors(bb));
          plot(t, x_gamma(bb,:),            plot_colors(bb), 'LineWidth', 1.25);
        end
        xlabel('Time [s]'), ylabel('State [-]'), hold off
        legend('x','x+z','P(x+z)'), title(sprintf('State  (iteration %i,%i)', ii, jj))
        
        % Plot inputs (old, new, projection)
        t = linspace(0, t_f, N); t = t(1:end-1);
        subplot(2,3,2), cla, grid on, hold on
        for bb=1:m
          plot(t, u(bb,:),                 [plot_colors(bb) '-.']);
          plot(t, u(bb,:) + gamma*v(bb,:), plot_colors(bb));
          plot(t, u_gamma(bb,:),            plot_colors(bb), 'LineWidth', 1.25);
        end
        xlabel('Time [s]'), ylabel('Input [-]'), hold off
        legend('u','u+v','P(u+v)'), title(sprintf('Input  (iteration %i,%i)', ii, jj))

         % Plot step in states
         t = linspace(0, t_f, N);
         subplot(2,3,4), cla, grid on, hold on
         for bb=1:n
           plot(t, gamma*z(bb,:), plot_colors(bb));
           legend_state_string{bb} = sprintf('x_%i', bb);
         end
         xlabel('Time [s]'), ylabel('State [-]'), hold off
         legend(legend_state_string), title(sprintf('Change in state  (iteration %i,%i)', ii, jj))
         
         % Plot step in inputs
         t = linspace(0, t_f, N); t = t(1:end-1);
         subplot(2,3,5), cla, grid on, hold on
         for bb=1:m
           plot(t, gamma*v(bb,:), plot_colors(bb));
           legend_input_string{bb} = sprintf('u_%i', bb);
         end
         xlabel('Time [s]'), ylabel('Input [-]'), hold off
         legend(legend_input_string), title(sprintf('Change in input  (iteration %i,%i)', ii, jj))
         
         % Compute cost as function of step length
         gamma_vec=-0.25:0.025:1;
         xx_unp = zeros(n,N); xx_unp(:,1) = x(:,1);
         uu_unp = zeros(m,N-1);
         xx_prj  =zeros(n,N); xx_prj(:,1) = x(:,1);
         uu_prj = zeros(m,N-1);
         desc1_temp = []; desc2_temp = 0;
         for bb = 1:length(gamma_vec)
           for cc=1:N-1
             uu_unp(:,cc) = (u(:,cc) + gamma_vec(bb)*v(:,cc)) ;
             xx_unp(:,cc+1) = dynamics(h_dyn, xx_unp(:,cc), uu_unp(:,cc), dyn_params, hh);
             
             uu_prj(:,cc) = (u(:,cc) + gamma_vec(bb)*v(:,cc)) + K_prj(:,:,cc)*((x(:,cc) + gamma_vec(bb)*z(:,cc)) - xx_prj(:,cc));
             xx_prj(:,cc+1) = dynamics(h_dyn, xx_prj(:,cc), uu_prj(:,cc), dyn_params, hh);
           end
           costs_unp(bb) = costfun(h_cost, h_con, xx_unp, uu_unp, x_des, u_des, cost_params, delta, epsilon);
           costs_prj(bb) = costfun(h_cost, h_con, xx_prj, uu_prj, x_des, u_des, cost_params, delta, epsilon);
         end
         
         % Plot descent in cost
         if jj == 1
           subplot(2,3,3), cla, grid on, hold on
             plot(gamma_vec, c + gamma_vec*desc1, 'Color', [0.7 0.7 0.7], 'LineWidth', 1.5);
             plot(gamma_vec, costs_unp, 'k');
             plot(gamma_vec, costs_prj, 'k--');
             plot(gamma,c_gamma,'o',0,c,'*','MarkerFaceColor',[0 0 0],'MarkerSize',5)
             h = fill([0, 0, 1, 1], [c + desc1, c, c + alpha*desc1, c + desc1], 'b');
             set(h,'FaceAlpha',0.1);
             set(h,'EdgeColor','None');
             xlabel('Step length [-]'), ylabel('Cost [-]'), hold off
             legend('approx (quad)', 'real (without PO', 'real (with PO)', 'Step accepted')
             title(sprintf('Change in cost  (iteration %i,%i)', ii, jj))
             ylim auto
           try
             limits = ylim;
             lim_high = limits(1);  lim_low = limits(2);
             limits_diff = lim_high - lim_low;
             ylim([max([0,  c + desc1 - limits_diff/4,  lim_low]), ...
               min(c + limits_diff/4,  lim_high)])
           catch
             % Do nothing
           end
           
           subplot(2,3,6), cla, grid on, hold on
             plot(gamma_vec, c + gamma_vec*desc1 + gamma_vec.^2*desc2, 'Color', [0.7 0.7 0.7], 'LineWidth', 1.5);
             plot(gamma_vec, costs_unp, 'k');
             plot(gamma_vec, costs_prj, 'k--');
             plot(gamma,c_gamma,'o',0,c,'*','MarkerFaceColor',[0 0 0],'MarkerSize',5)
             h = fill([0, 0, 1, 1], [c, c + desc1, c + desc1, c + alpha*desc1], 'b');
             set(h,'FaceAlpha',0.1);
             set(h,'EdgeColor','None');
             xlabel('Step length [-]'), ylabel('Cost [-]'), hold off
             legend('approx (quad)', 'real (without PO', 'real (with PO)')
           title(sprintf('Change in cost  (iteration %i,%i)', ii, jj))
             
         else
           subplot(2,3,3), hold on
             plot(gamma, c_gamma, 'o','MarkerFaceColor',[0 0 0],'MarkerSize',5)
             title(sprintf('Change in cost  (iteration %i,%i)', ii, jj))
             hold off
             
           subplot(2,3,6), hold on
             plot(gamma, c_gamma, 'o','MarkerFaceColor',[0 0 0],'MarkerSize',5)
             title(sprintf('Change in cost  (iteration %i,%i)', ii, jj))
           hold off
         end
                 
         pause()
      end % END(if plot_intermediates)
            
      % Armijo's rule            
      %
      % AS 26 Aug 2017 - I have removed the check desc1 < 0 as iterations
      % have to be stopped before if that happens  
      % if ((c_gamma < c + gamma*(alpha*desc1)) && (desc1 < 0)) || (-desc1 < AbsTol) % -desc1<AbsTol avoid extra computations
      %
      if c_gamma < c + gamma*(alpha*desc1) 
        break; % exit back tracking
      end
      
      gamma = beta*gamma; % back track based on value of beta (< 1)
    end
    % -----------------------------------------------------------------
    
    % ---- Processing -------------------------------------------------
    
    % 26 Aug 2017 The following comment is nonsense
    % if no valid iteration has been found, it is better to abort
    % 
    % If no step is found, set descent identically zero and keep the
    % 'old' trajectory. Else update to new trajectory
    % 
    
    if jj == maxStepLenghtIter
      % descent = 0;
      error('no discent found during backtracking line search');
    else
      x       = x_gamma; 
      u       = u_gamma; 
      descent = -desc1;
    end
    
    figure(10000)
      if ii == 1 
        clf 
      end 
      hold on
      plot(ii,log(descent),'+')
      grid on, zoom on
    figure(1)
    
    % If enabled, print diagnostic information
    [c, ~, c_con] = costfun(h_cost, h_con, x, u, x_des, u_des, cost_params, delta, epsilon);
    print_diagnostics_iteration(print_diagnostics, ii, c, c_con, descent, gamma, isPD_count, delta, epsilon);
    
%     if plot_intermediates
      % AS Jun 16, 2017 Plot the current iterate around the sphere
      figure(1000)
        hold on, grid on
        plot3(x(1,:),x(3,:),x(5,:),'linewidth',2);
        
      %figureEND  
%     end
    
    % Store detailed information in struct.
    output{end+1}.x = x;
    output{end}.u = u;
    output{end}.z = z;
    output{end}.v = v;
    output{end}.cost = c;
    output{end}.descent = descent;
    output{end}.gamma = gamma;
    output{end}.delta = delta;
    output{end}.epsilon = epsilon;
    
    % If all constraints are already in the logaritmic part, delta can
    % be reduced instantly without affecting the cost. 
    % Delta is set to (virtually) zero and isDeltaSetToMin is set to TRUe
    
    if isa(h_con, 'function_handle')
      % compute min value of the constraints
      [~, ~, ~, cval_min] = costfun(h_cost, h_con, x, u, x_des, u_des, cost_params, delta, epsilon);
      % check if the current trajectory (x,u) is delta-feasible, as long as
      % delta is not already set to its minimum
      if (cval_min > delta) && ~isDeltaSetToMin 
        delta = eps; % the deltabeta function does NOT work for delta = 0 
                     % (I got Inf as result) therefore we set delta to 
                     % eps (the smallest value in MATLAB)
                     
        isDeltaSetToMin = true; % flag delta-feasibility has been reached
      end
    end
        
    % Check if descent is smaller than AbsTol (same as in PRONTO)
    if descent < AbsTol 
      break;
    end
    
    % -----------------------------------------------------------------
  end
   
  %%%%%%% dPRONTO iterations (END) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
  % We now have to reduce epsilon (this does not affect delta-feasibility)
  % We have also to update delta, in case we do not have delta feasibility
  % if c is < 0 (and we want c > 0), the trajectory is unfeasible
  % and this should be reported (large epsilon and delta non too small are
  % supposed to push the initial trajectory in the interior)
  
  % AS 26 Aug 2017 - The following command is removed as this seemed useless...
  % isKvCompleted = false; 
  
  % Delta and epsilon update rules
  % 
  % FIRST goal should be to get a feasible trajectory
  % SECOND goal is to reduce delta as in a standard continuation method
  % To this end, once found a feasible trajectory,  et delta to
  % (virtually) zero, to get a full primal interior point method 
  %
  % 26 Aug 2017 - I have completely change the logic Remon had
  %               to use the standard one in PRONTO
  %
  
  if isa(h_con, 'function_handle') % run only if there are constraints
    if isDeltaSetToMin % we are running a fully interior point method
      epsilon = epsilon_step * epsilon; % reduce epsilon (path following)
    else
      error('dPRONTO failed to find a feasibilty solution')
      % to be changed into something milder, or different strategy (like
      % INCREASING epsilon) to get a feasibile trajectory
    end
  else
    break % if there are no constraints, PRONTO has finished
  end
  
  if epsilon < epsilon_min % termination condition for the continuation 
    break 
  end
    
  % +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
end

print_diagnostics_footer(print_diagnostics);
% *************************************************************************
