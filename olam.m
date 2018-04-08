clc; clear;

dados = load('C:\Users\Raul Melo\Documents\OLAM\dataIris.dat');
% dados = load('C:\Users\Raul Melo\Documents\OLAM\coluna_vertebral.dat');
% dados = load('C:\Users\Raul Melo\Documents\OLAM\dermatologia.dat');
numRealizacoes = 20;
numAtributos = 4;                   % Iris 4, coluna 6, derm 34
numNeuronioSaida = 3;               % Iris 3, coluna 3, derm 6

dados = [normaliza(dados(:,1:numAtributos)) dados(:,numAtributos+1:numAtributos+numNeuronioSaida)];

numPadroes = size(dados,1);

for k=1:numRealizacoes
    dados = dados(randperm(numPadroes),:);
    [X_treino, Y_treino, X_teste, Y_teste] = separaDados(dados, numAtributos, numNeuronioSaida, 0.8);
    X_treino = [-ones(size(X_treino,1),1) X_treino];
    X_teste = [-ones(size(X_teste,1),1) X_teste];

    W = X_treino\Y_treino;
% ()
    count = 0;
    for i = 1: size(Y_teste,1)
        saida = binariza(logsig(X_teste(i,:)*W));
        if(isequal(saida,Y_teste(i,:)))
            count = count + 1;
        end
    
    end
    acc(k) = 100*(count/size(Y_teste,1));
end

media_acc = mean(acc)
desvio_padrao = std(acc)



