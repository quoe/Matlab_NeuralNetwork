%ConverValues2Categories 
%ind2vec (Convert indices to vectors)
Tc = T;
Tu = unique(Tc); % Уникальные значения
numTu = max(Tu, 2); % Количество уникальных значений
Tmax = max(Tu);
Tvec = full(ind2vec(Tc, Tmax));