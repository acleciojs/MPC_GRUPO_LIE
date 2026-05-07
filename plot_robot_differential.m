function plot_robot_differential(H, scale)

% PLOTA UM ROBÔ DIFERENCIAL NO PLANO (OTIMIZADO PARA LOOP)

if nargin < 2
    scale = 0.02;
end

% =========================================================
% OBJETOS PERSISTENTES (NÃO RECRIA A CADA LOOP)
% =========================================================
persistent h_body h_center h_arrow traj_x traj_y initialized

if isempty(initialized)

    figure(1); clf; hold on; grid on; axis equal

    h_body   = fill(nan, nan, 'b');
    h_center = plot(nan, nan, 'ro', 'LineWidth',2);
    h_arrow  = quiver(0,0,0,0,'r','LineWidth',2,'MaxHeadSize',2);

    traj_x = [];
    traj_y = [];

    initialized = true;

    xlim([-10 10])
    ylim([-10 10])
end

% =========================================================
% Extrair pose
% =========================================================
x = H(1,3);
y = H(2,3);
R = H(1:2,1:2);

e1 = R(:,1);
e2 = R(:,2);

% =========================================================
% Geometria do robô
% =========================================================
p_front = [x; y] + scale*e1;
p_left  = [x; y] - 0.5*scale*e1 + 0.5*scale*e2;
p_right = [x; y] - 0.5*scale*e1 - 0.5*scale*e2;

X = [p_front(1), p_left(1), p_right(1)];
Y = [p_front(2), p_left(2), p_right(2)];

% =========================================================
% UPDATE (AO INVÉS DE RECRIAR)
% =========================================================
set(h_body, 'XData', X, 'YData', Y);
set(h_center, 'XData', x, 'YData', y);

set(h_arrow, ...
    'XData', x, ...
    'YData', y, ...
    'UData', e1(1)*scale*1.5, ...
    'VData', e1(2)*scale*1.5);

% trajetória
traj_x(end+1) = x;
traj_y(end+1) = y;

plot(traj_x, traj_y, 'k--')

drawnow limitrate

end