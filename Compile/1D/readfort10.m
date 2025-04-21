close all;
clear;clc;

TitleName = 'Compile1D-T100{\mum}'; 

fid = fopen('fort.10');
dataAll= textscan(fid,'%s');
fclose(fid); clear fid;
dataListAll=dataAll{1,1};

countDataAll = length(dataListAll);

countS1 = str2num(dataListAll{1,1});
countS2 = str2num(dataListAll{countS1*3+2,1});

iBeginS1 = 2;
iEndS1   = countS1*3+1;
iBeginS2 = iEndS1+2;
iEndS2   = countDataAll;

columnsS1 =3;
columnsS2 =(countDataAll-1-countS1*3-1)/countS2;

dataListS1 = dataListAll(iBeginS1:iEndS1,1);
dataListS2 = dataListAll(iBeginS2:iEndS2,1);

dataArrayS2 = reshape(dataListS2,countS2,columnsS2);

clear dataAll dataListAll dataListS1 dataListS2;

time    = str2double(dataArrayS2(strmatch('TIME',dataArrayS2,'exact'),3:columnsS2));
xc      = str2double(dataArrayS2(strmatch('XC',dataArrayS2,'exact'),3:columnsS2));
Tipro   = str2double(dataArrayS2(strmatch('TI',dataArrayS2,'exact'),3:columnsS2));
Tepro   = str2double(dataArrayS2(strmatch('TE',dataArrayS2,'exact'),3:columnsS2));
Trpron  = str2double(dataArrayS2(strmatch('TR',dataArrayS2,'exact'),3:columnsS2));
Rhopro  = str2double(dataArrayS2(strmatch('R',dataArrayS2,'exact'),3:columnsS2));
Ptpro   = str2double(dataArrayS2(strmatch('PT',dataArrayS2,'exact'),3:columnsS2));
Nepro   = str2double(dataArrayS2(strmatch('DENE',dataArrayS2,'exact'),3:columnsS2));
x       = str2double(dataArrayS2(strmatch('X',dataArrayS2,'exact'),3:columnsS2));
v       = str2double(dataArrayS2(strmatch('V',dataArrayS2,'exact'),3:columnsS2));
LPower  = str2double(dataArrayS2(strmatch('LPOWER',dataArrayS2,'exact'),3:columnsS2));
rhorDT  = str2double(dataArrayS2(strmatch('RHORDT',dataArrayS2,'exact'),3:columnsS2));
Vimp    = str2double(dataArrayS2(strmatch('VIMPLO',dataArrayS2,'exact'),3:columnsS2));
alpha   = str2double(dataArrayS2(strmatch('ALPHA',dataArrayS2,'exact'),3:columnsS2));

n       = size(xc,1);
Trpro(1:n,:)     = (Trpron(1:n,:)+Trpron(2:n+1,:))/2;

xc      = xc*1e4;             % convert to micrometer
x       = x*1e4;              % convert to micrometer
time    = time*1e9;           % convert to ns
LPower  = LPower*1e-7;        % convert to Watt
Nepro   = Nepro/1.75e21;      % normalize to critical density 

ntime = length(time);
xnc   = time-time;
for i = 1:ntime
    index = find(-v(:,i)<0,1);
    index(index<62)=200;
    xnc(i)= x(index,i);
end
% return;

figure
set(gca,'fontsize',14);
box on;
yyaxis left;
hold on;
for i=1:1:size(x,1)
     if(i<=61)
         % h1=plot(time,x(i,:),'g-');
     elseif(i<=181)
         h1=plot(time,x(i,:),'-');
     else
         h1=plot(time,x(i,:),'k-');
     end
end
% xlim([0 5.0]);
ylim([0 700]);
xlabel('Time(ns)');
ylabel('Radius(\mum)');


yyaxis right;
LPower(LPower<1e-12) = 1e-12;
h3 = plot(time,LPower/5.6*1e-12,'*-');
xlim([0 8.0]);


% set(gca,'yscale','log');
ylabel('Laser(TW)');
title(TitleName)
% set(gca,'ylim',[10^10,10^16],'yTick',10.^(8:2:40));
% ylim([0 10]);
% set(gca,'Ycolor','k','Yminortick','off');
% return;

%%
JJ_gas  = 60;
JJ_fuel = 120+JJ_gas;
JJ_Be   = 60+JJ_fuel;
Tipro_ave = sum(Tipro(JJ_fuel-120+1:JJ_fuel,:),1)*1e-3/120;
Rhopro_ave = sum(Rhopro(JJ_fuel-120+1:JJ_fuel,:),1)*1e0/120;

figure
hold on;
box on;
set(gca,'fontsize',14);
yyaxis left;
plot(time,Rhopro_ave,'*-');      %averaged value
xlim([4.5 7.5]);
ylim([0 150]);
xlabel('Time (ns)');
ylabel('Density(g/cc)');

yyaxis right;
plot(time,rhorDT,'r.-');      
plot(time,Tipro_ave,'x-');      %averaged value
ylabel('\rhoR(g/cm^2)/T(keV)');
ylim([0 0.8]);

legend('Density','Areal Density','Temperature',Location='northeast');
title(TitleName)

%%
% figure
% hold on;
% plot(Rhopro_ave,Tipro_ave*1e3,'*-')
% plot(Rhopro_ave,14.05*Rhopro_ave.^(2/3),'r*-')
% set(gca,'xscale','log');
% set(gca,'yscale','log');
% xlim([1e0,1.5e2]);
% ylim([1e1,5e2]);
% xlabel('Density(g/cc)');
% ylabel('T(eV)');

%%
% figure
% plot(time,1e-3*Rhopro_ave.^2.*Tipro_ave.^0.5,'*-')
% xlim([1e0,1.5e2]);
% xlabel('Time')
% ylabel('Power')

%%
% return;
% figure
% yyaxis left;
% plot(time,LPower/5.6/8*1e-12,'*-');
% ylabel('Power per Beam (TW)');
% yyaxis right;
Lenergy=cumtrapz(time,LPower/5.6*1e-12);
% plot(time,Lenergy,'x-')
% xlabel('Time (ns)');
% ylabel('Laser Energy (kJ)');
% xlim([0 8]);
% title(TitleName)

% return;

%%
figure
set(gca,'fontsize',14);
box on;
yyaxis left;
hold on;
ifar= x(61,:)./(x(180,:)-x(61,:));
% ifar= 0.5*(x(61,:)+1*xnc)./(xnc-x(61,:));
% plot(time,ifar,'x-');
ylabel('IFAR');
plot(time,LPower/5.6/8*1e-12,'*-');
h1=plot(time,x(120,:)*1e-3,'r.-');
ylabel('Laser(TW)/Radius(mm)');

% ylim([0,40]);
yyaxis right;
v1=v;
v1(v>0) =0 ;
Vimp= sum(v1(JJ_fuel-120+1:JJ_fuel,:),1)*1e-0/120;
plot(time,-Vimp*1e-5,'x-');
xlim([0,8]);
xlabel('Time (ns)');
ylabel('V (km/s)');
title(TitleName)

% figure
% plot(time,alpha,'x-');
% xlim([0,7]);
% xlabel('Time (ns)');
% ylabel('\alpha');
% set(gca,'fontsize',14);

% fprintf("alpha   = %.4f\n",max(alpha))
fprintf("rhoRmax = %.4f g/cm^2\n",max(rhorDT))
fprintf("Rhomax  = %.4f g/cc\n",max(Rhopro_ave))
fprintf("Timax   = %.4f keV\n",max(Tipro_ave))
fprintf("Vimplo  = %.4f km/s\n",max(-Vimp*1e-5))
fprintf("IFAR    = %.4f\n",max(ifar))
fprintf("Lenergy = %.4f kJ\n",max(Lenergy))

return;
figure
h=surf(1:1:size(Rhopro,1),time,(Rhopro'));
set(h,'EdgeAlpha',0.5)
xlim([58 160])
ylim([0 5])
zlim([0 10])
colorbar
caxis([0 10])
ylabel('Time(ns)')
xlabel('Cell Number')
zlabel('Density(g/cc)')


return;

set(gcf, 'PaperUnits', 'inches', 'PaperSize', [4 6]);
print(gcf,'Traj_Laser','-dpng','-r300')
