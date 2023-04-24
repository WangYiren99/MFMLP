function createfigure(X1, YMatrix1, Corr)
%CREATEFIGURE(X1, YMatrix1)
%  X1:  x ���ݵ�����
%  YMATRIX1:  y ���ݵľ���

%  �� MATLAB �� 19-Feb-2022 11:12:35 �Զ�����

% ���� figure
% figure1 = figure('PaperOrientation','landscape',...
%     'PaperSize',[29.69999902 20.99999864]);
% figure1 = figure('PaperOrientation','landscape',...
%     'PaperSize',[10 10]);
figure1=figure();


% ���� axes
axes1 = axes('Parent',figure1,'Position',[0.1 0.1 0.5 0.7]);
hold(axes1,'on');

% ʹ�� plot �ľ������봴������
plot1 = plot(X1,YMatrix1,'MarkerSize',8,'Marker','.','LineWidth',1,...
    'Parent',axes1);
set(plot1(1),'DisplayName',' Ground Truth','Color',[124 187 0]/255);
set(plot1(2),'DisplayName',' lambdaNet, corr: '+string(roundn(Corr(1),-4)),'Color',[0 161 241]/255);
set(plot1(3),'DisplayName',' TSA-Net, corr: '+string(roundn(Corr(5),-4)),'Color',[255 187 0]/255);
set(plot1(4),'DisplayName',' DGSMP, corr: '+string(roundn(Corr(6),-4)),'Color',[0 0.45 0.74]);
set(plot1(5),'DisplayName',' MST-L, corr: '+string(roundn(Corr(4),-4)),'Color',[0.85 0.33 0.1]);
set(plot1(6),'DisplayName',' CST-L*, corr: '+string(roundn(Corr(2),-4)),'Color',[0.49 0.18 0.56]);
set(plot1(7),'DisplayName',' DAUHST-9stg, corr: '+string(roundn(Corr(3),-4)),'Color',[71 51 53]/255);
set(plot1(8),'DisplayName',' MFMLP-9stg, corr: '+string(roundn(Corr(7),-4)),'Color',[252 41 30]/255);
% set(plot1(9),'DisplayName',' MFMLP-11stg, corr: '+string(roundn(Corr(8),-4)),'Color',[246 83 20]/255);
% set(plot1(10),'DisplayName',' MFMLP-plus, corr: '+string(roundn(Corr(9),-4)),'Color',[189 30 30]/255);




% ȡ�������е�ע���Ա����������� Y ��Χ
ylim(axes1,[0 1]);
% xlim(axesl,[0 1])
box(axes1,'on');
hold(axes1,'off');
% ������������������
set(axes1,'FontName','Arial','FontSize',10,'LineWidth',1);

% ���� ylabel
ylabel('Density','FontSize',13,'FontName','Arial');

% ���� xlabel
xlabel('Wavelength (nm)','FontSize',12,'FontName','Arial');
% ���� legend
legend1 = legend(axes1,'show');
set(legend1,...
    'Position',[0.25 0.15 0.187369795342287 0.36915888702758],...
    'FontSize',12,...
    'EdgeColor',[1 1 1]);
