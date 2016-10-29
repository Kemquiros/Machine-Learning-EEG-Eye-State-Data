%{
    Copyright 2015 John Tapias Zarrazola

    This file is part of Machine-Learning-EEG-Eye-State.

    Machine-Learning-EEG-Eye-State is free software: you can redistribute it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Machine-Learning-EEG-Eye-State in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with Machine-Learning-EEG-Eye-State.  If not, see <https://www.gnu.org/licenses/lgpl.txt>.

Licence: LGPL - 3.0
Software Developers: 
-John Tapias Zarrazola
-Esteban Catano Escobar
Date:10th October / 2015

Database:
https://archive.ics.uci.edu/ml/datasets/EEG+Eye+State


%}

clc
clear all
% close all

load('Data.mat');

X=Data(:,1:6);
Y=Data(:,end);

NumMuestras=size(X,1); 
Rept=10;

EficienciaTest=zeros(1,Rept);

punto=input('Ingrese 3 para Mezcla de Gaussianas, 4 para arboles de decisión, ó 5 para Random Forest: ');

if punto==3
    
    %%% punto GMM %%%
     
    NumClases=length(unique(Y)); %%% Se determina el número de clases del problema. 
    Mezclas=1; %%% Se determina el número de Gaussianas que tiene el modelo.
    
    for fold=1:Rept

        %%% Se hace la partición de las muestras %%%
        %%%      de entrenamiento y prueba       %%%
        
        rng('default');
        particion=cvpartition(NumMuestras,'Kfold',Rept);
        indices=particion.training(fold);
        Xtrain=X(particion.training(fold),:);
        Xtest=X(particion.test(fold),:);
        Ytrain=Y(particion.training(fold));
        Ytest=Y(particion.test(fold));

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %%% Se normalizan los datos %%%

        [XtrainNormal,mu,sigma] = zscore(Xtrain);
        XtestNormal = (Xtest - repmat(mu,size(Xtest,1),1))./repmat(sigma,size(Xtest,1),1);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%% Entrenamiento de los modelos. Recuerde que es un modelo por cada clase. %%%
        
        vInd=(Ytrain == 1);
        XtrainC1 = Xtrain(vInd,:);
        if ~isempty(XtrainC1)
            Modelo1=entrenarGMM(XtrainC1,Mezclas);
        else
            error('No hay muestras de todas las clases para el entrenamiento');
        end
        
        vInd=(Ytrain == 2);
        XtrainC2 = Xtrain(vInd,:);
        if ~isempty(XtrainC2)
            Modelo2=entrenarGMM(XtrainC2,Mezclas);
        else
            error('No hay muestras de todas las clases para el entrenamiento');
        end
        
        vInd=(Ytrain == 3);
        XtrainC3 = Xtrain(vInd,:);
        if ~isempty(XtrainC3)
            Modelo3=entrenarGMM(XtrainC3,Mezclas);
        else
            error('No hay muestras de todas las clases para el entrenamiento');
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%% Validación de los modelos. %%%
        
        probClase1=testGMM(Modelo1,Xtest);
        probClase2=testGMM(Modelo2,Xtest);
        probClase3=testGMM(Modelo3,Xtest);
        Matriz=[probClase1,probClase2,probClase3];
        
        [~,Yest] = max(Matriz,[],2);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        MatrizConfusion = zeros(NumClases,NumClases);
        for i=1:size(Xtest,1)
            MatrizConfusion(Yest(i),Ytest(i)) = MatrizConfusion(Yest(i),Ytest(i)) + 1;
        end
        EficienciaTest(fold) = sum(diag(MatrizConfusion))/sum(sum(MatrizConfusion));
        
    end
    
    Eficiencia = mean(EficienciaTest);
    IC = std(EficienciaTest);
    Texto=['La eficiencia obtenida fue = ', num2str(Eficiencia),' +- ',num2str(IC)];
    disp(Texto);

    %%% Fin punto GMM %%%
    
elseif punto==4
    
    %%% punto Arboles de decisión %%%
    
    NumClases=length(unique(Y)); %%% Se determina el número de clases del problema. 
    
    for fold=1:Rept

        %%% Se hace la partición de las muestras %%%
        %%%      de entrenamiento y prueba       %%%
        
        rng('default');
        particion=cvpartition(NumMuestras,'Kfold',Rept);
        indices=particion.training(fold);
        Xtrain=X(particion.training(fold),:);
        Xtest=X(particion.test(fold),:);
        Ytrain=Y(particion.training(fold));
        Ytest=Y(particion.test(fold));

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %%% Se normalizan los datos %%%

        [XtrainNormal,mu,sigma] = zscore(Xtrain);
        XtestNormal = (Xtest - repmat(mu,size(Xtest,1),1))./repmat(sigma,size(Xtest,1),1);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%% Entrenamiento de los modelos. Recuerde que es un modelo por cada clase. %%%

        Modelo=entrenarTREE(Xtrain,Ytrain);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%% Validación de los modelos. %%%
        
        NivelPoda=2;
        
        Modelo=prune(Modelo,'level',NivelPoda);
        Yest=testTREE(Modelo,Xtest);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        MatrizConfusion = zeros(NumClases,NumClases);
        for i=1:size(Xtest,1)
            MatrizConfusion(Yest(i),Ytest(i)) = MatrizConfusion(Yest(i),Ytest(i)) + 1;
        end
        EficienciaTest(fold) = sum(diag(MatrizConfusion))/sum(sum(MatrizConfusion));
        
    end
    
    Eficiencia = mean(EficienciaTest);
    IC = std(EficienciaTest);
    Texto=['La eficiencia obtenida fue = ', num2str(Eficiencia),' +- ',num2str(IC)];
    disp(Texto);

    %%% Fin punto Arboles de decisión %%%
    
elseif punto==5
    
    %%% punto Random Forest %%%
    
    NumClases=length(unique(Y)); %%% Se determina el número de clases del problema.
    
    for fold=1:Rept

        %%% Se hace la partición de las muestras %%%
        %%%      de entrenamiento y prueba       %%%
        
        rng('default');
        particion=cvpartition(NumMuestras,'Kfold',Rept);
        indices=particion.training(fold);
        Xtrain=X(particion.training(fold),:);
        Xtest=X(particion.test(fold),:);
        Ytrain=Y(particion.training(fold));
        Ytest=Y(particion.test(fold));

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %%% Entrenamiento de los modelos. Recuerde que es un modelo por cada clase. %%%

        NumArboles=500;
        Modelo=entrenarFOREST(NumArboles,Xtrain,Ytrain);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%% Validación de los modelos. %%%
        
        Yest=testFOREST(Modelo,Xtest);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        MatrizConfusion = zeros(NumClases,NumClases);
        for i=1:size(Xtest,1)
            MatrizConfusion(Yest(i),Ytest(i)) = MatrizConfusion(Yest(i),Ytest(i)) + 1;
        end
        EficienciaTest(fold) = sum(diag(MatrizConfusion))/sum(sum(MatrizConfusion));
        
    end
    
    Eficiencia = mean(EficienciaTest);
    IC = std(EficienciaTest);
    Texto=['La eficiencia obtenida fue = ', num2str(Eficiencia),' +- ',num2str(IC)];
    disp(Texto);

    %%% Fin punto Random Forest %%%

end



