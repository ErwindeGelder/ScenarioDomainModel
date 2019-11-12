% function to get OpenDRIVE parameters from a gpx-file
% gpx-file created with GPSvisualizer
% (http://www.gpsvisualizer.com/elevation) based on google maps trajectory url
% [OD] = create_trajectory_slope(GPXfile)
% OD, Struct with OpenDRIVE parameter information regarding trajectory and elevation polynomials
% filename, name of gpx-file
% pieces, number of pieces for spine fitting
% Jeroen Broos, TNO, May, 2018

function [OD] = create_trajectory_slope(GPXfile,pieces)

% load GPX data
[Lon, Lat, Ele] = readGPX(GPXfile);

% From GPS to X-Y coordinates
[X,Y] = fnc_GPS2XY(Lon,Lat);

%% Fit b-spline
% [x y] data
pp_gps = splinefit(X,Y,pieces,4);
step = 0.01;
XX = pp_gps.breaks(1):step:pp_gps.breaks(end);
if X(end) < 0
    XX = flip(XX);
end
YY = ppval(pp_gps,XX);

% Determine distance of x-y trajectory
D(1) = 0;
for i = 1:length(XX)-1
    D(i+1) = D(i) + sqrt((XX(i+1)-XX(i))^2+(YY(i+1)-YY(i))^2);
end
D = D';

% elevaltion profile wrt distance of trajectory in track coordinate system
E = interp1(X,Ele,XX);
pp_Ele = splinefit(D,E,pieces);
EE = ppval(pp_Ele,D);

%% Create output signals
ID_breaks = round((pp_gps.breaks-pp_gps.breaks(1))/step)+1;
ID_breaks(end) = ID_breaks(end)-1;
% trajectory data for OpenDRIVE poly3 description (y = a + b*x + c*x^2 + d*x^3)
OD.geo.X  = pp_gps.breaks(1:end-1)';                    % X-start position of polynominal
OD.geo.Y  = pp_gps.coefs(:,4);                          % Y-start position of polynominal
OD.geo.H  = atan(pp_gps.coefs(:,3));                    % Initial heading of polynominal
OD.geo.c  = pp_gps.coefs(:,2);                          % c-parameter of polynomonal
OD.geo.d  = pp_gps.coefs(:,1);                          % d-parameter of polynomonal
OD.geo.Ls = D(ID_breaks(1:end-1));                      % Start position of polynominal in s (distance of road)
OD.geo.L  = D(ID_breaks(2:end))-D(ID_breaks(1:end-1));  % Length of polynominal in s (distance of road)
OD.geo.XX = XX;                                         % X-position [m]
OD.geo.YY = YY;                                         % Y-position [m]
OD.geo.p  = pp_gps;

% elevalation data for OpenDRIVE (elev = a + b*ds + c*ds^2 + d*ds^3, ds = distance on OpenDrive road)
OD.ele.s  = pp_Ele.breaks(1:end-1)'; % s-start position of polynominal
OD.ele.a  = pp_Ele.coefs(:,4);       % a-parameter of polynomonal
OD.ele.b  = pp_Ele.coefs(:,3);       % b-parameter of polynomonal
OD.ele.c  = pp_Ele.coefs(:,2);       % c-parameter of polynomonal
OD.ele.d  = pp_Ele.coefs(:,1);       % d-parameter of polynomonal
OD.ele.D  = D;                       % Distance [m]
OD.ele.EE = EE;                      % Height [m]

%% Figures
figure(1)
subplot(1,2,1); hold on
p(1) = plot(X,Y,'bo');
p(2) = plot(XX(ID_breaks),YY(ID_breaks),'ro','LineWidth',3);
p(3) = plot(OD.geo.X,OD.geo.Y,'gx','LineWidth',3);
plot(X,Y,'b','LineWidth',3);
plot(XX,YY,'r','LineWidth',1);
axis square equal
xlabel('X-position [m]')
ylabel('Y-position [m]')
legend(p(1:2),'Google maps data','Spline fit')

subplot(1,2,2)
p(1) = plot(D,E,'LineWidth',3); hold on
p(2) = plot(D,EE,'r','LineWidth',1);
% axis square equal
xlabel('Distance [m]')
ylabel('Height [m]')
legend(p(1:2),'Google maps data','Spline fit')

