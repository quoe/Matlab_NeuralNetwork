%NAR_TSTestLoop_DrawNowForecast

figForecast = figure('Name','Прогноз');
for n = size(Target, 2)-round(size(Target, 2)/2):-1:0
    clf(figForecast) % Очистить график
    figure(figForecast); % График прогноза
    plot(Target); hold on;
    T = Target(1:end-n);
    if ~iscell(T) % Выход
        OUT = con2seq(T);
    else
        OUT = T;
    end
    y_cl = []; y_fit = [];
    y_mean = 0; % 0 - не выводить среднее, 1 - среднее графиков
    for i = 1:3%size(bestResultNetsSort, 2)
        net = bestResultNetsSort{i};
        [Xs,Xi,Ai,Ts] = preparets(net,{},{},OUT);
        %net = train(net,Xs,Ts,Xi,Ai);
        %view(net)
        [Y,Xf,Af] = net(Xs,Xi,Ai);
        perf = perform(net,Ts,Y);
        [netc,Xic,Aic] = closeloop(net,Xf,Af);
        %view(netc)
        %[Xs,Xi,Ai,Ts] = preparets(netc,{},{},OUT);
        y2 = netc(Xs,Xic,Aic); % y2 = netc(cell(1,150),Xic,Aic); cell - прогноз на количество значений
        y_cl = [y_cl; y2];
        y_fit = [y_fit; Y];

        %plot(cell2mat(OUT)); hold on; plot(cell2mat(y2));
        %plot(cell2mat(y2));
        plot(cell2mat(OUT), 'DisplayName', [sprintf('%i) ', size(T, 2)) ' Target'], 'LineWidth', 3); hold on; % График текущей наблюдаемой части прогноза
        if y_mean == 0 % Отдельные графики
            figure(figForecast); % График прогноза
            if i == 1 % Первый наилучший график толстой линией
                plot(cell2mat([OUT cell2mat(y2)]), 'DisplayName', [sprintf('%i) ', i) net.userdata], 'LineWidth', 2); hold on; 
            else 
                plot(cell2mat([OUT cell2mat(y2)]), 'DisplayName', [sprintf('%i) ', i) net.userdata]); hold on; 
            end
        end
    end 
    drawnow
    %pause(3)
end
    