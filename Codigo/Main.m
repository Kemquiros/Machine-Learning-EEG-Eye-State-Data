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

clear all;
clc;

%archivo= csvread('./Datos/EEG-Eye-State.csv');
%[filasReales,columnasReales]=size(archivo)
load('./Datos/EEGEyeStateTest.mat');
[filas,columnas]=size(X);

%{
%Conocer el indice correcto de la primera muestra de X en el archivo real
vector = X(1,:);
existe=0;
apuntadorCorrecto=0;
vector=sort(vector);
for i=1:filasReales	
		if(vector==sort(archivo(i,1:end-1)))
			apuntadorCorrecto=i;
			existe=existe+1;						
		end	
end
existe
apuntadorCorrecto
%}

%Encontrar el orden correcto de las caracteristicas
%{
indices = zeros(1,columnas);
for i=1:columnas
	for j=1:columnas
		if(vector(1,i)==archivo(13498,j))
			indices(1,i)=j;
			break;
		end
	end
end
indices
%}

%Orden correcto de X respecto al archivo real
%7     3     2    10     1     6     5    11     4    12     8     9    14    13
%indices=[7  ,   3 ,    2 ,   10 ,    1  ,   6 ,    5 ,   11 ,    4 ,   12
%,    8 ,    9  ,  14  ,  13];
%{
%Ordenar la matriz con sus indices reales
indices=[7  ,   3 ,    2 ,   10 ,    1  ,   6 ,    5 ,   11 ,    4 ,   12 ,    8 ,    9  ,  14  ,  13];
Xordenado = zeros(filas,columnas);
for i=1:filas
	for j=1:columnas
		Xordenado(i,indices(j))=X(i,j);
	end
end
X=Xordenado;
%}
%--------------------PROGRAMA----------------------------------------------

numeroMuestrasClase1=sum(Y==0);
numeroMuestrasClase2=sum(Y==1);
disp('***Programa para detectar apertura o cierre del ojo humano***');
disp(strcat('Muestras de la clase 1 : ',num2str(numeroMuestrasClase1),', con un porcentaje de : ',num2str((numeroMuestrasClase1/length(Y))*100),'%'));
disp(strcat('Muestras de la clase 2 : ',num2str(numeroMuestrasClase2),', con un porcentaje de : ',num2str((numeroMuestrasClase2/length(Y))*100),'%'));
disp(strcat('Para un total de muestras : ',num2str(length(Y))));
input('\nHay un balance entre las muestras\n(Presione Enter)');
NumClases=length(unique(Y));
clc;
seleccion = input('Ingrese:\n(1) Regresión Logística\n(2) SVM\n(3) Mezcla de Discriminantes Gaussianas \n(4) K vecinos\n(5) RNA\n(6) Random Forest\n(7) Arbol de decisón\n(8) Ventana de Parzen\n(9) Reducción de dimensión\n(10) Predecir\n----> ');
%seleccion=1;

if seleccion == 1 %Regresión logística
    fold=10;
    EficienciaTest=zeros(1,fold);
    SensibilidadTest = zeros(1,fold);
    EspecificidadTest = zeros(1,fold);
    
    %--------------------Selección del grado del polinomio
    
    %grado=input('\nIngrese el grado del polinomio para la Regresión Logística:\n----> ');
    %eta=input('\nIngrese la tasa de aprendizaje para la Regresión Logística:\n----> ');
    eta = 0.1;
    grado = 5;
    X=potenciaPolinomio(X,grado);
    
  for i=1:fold  
    %--------------------Divisón de las muestras        
    %{
    Xentrenamiento: muestras para entrenar el sistema
    Yentrenamiento: salida del sistema para entrenar
    Xprueba: muestra para probar el sistema
    Yentrenamiento: salida del sistema para validar
     
    %}
    %rng('default');%Semilla aleatoria
    ind=randperm(filas); %%% Se seleccionan los indices de forma aleatoria
    tope=filas*0.9;%Se entrena con el 90% de las muestras
    Xentrenamiento=X(ind(1:tope),:);
    Xprueba=X(ind(tope+1:end),:);    
    Yentrenamiento=Y(ind(1:tope),:);
    Yprueba=Y(ind(tope+1:end),:);
    
    
    %-----------------Normalización        
    [Xentrenamiento,mu,sigma]=zscore(Xentrenamiento);
    Xprueba=normalizar(Xprueba,mu,sigma);
    
    
    %---------------Se extienden las matrices con el termino independiente          
    Xentrenamiento=[Xentrenamiento,ones(tope,1)];
    Xprueba=[Xprueba,ones(filas-tope,1)];%Columnas-Tope es la cantidad de prueba
    
    
    %----------------Se determinan los pesos de la regresión logística    
    W=regresionLogistica(Xentrenamiento,Yentrenamiento,eta); %%% Se optienen los W coeficientes del polinomio
    
    %--------------Eficiencia de la prueba


    Yestimado=(W*Xprueba')';
    %ECM = (sum((Yestimado-Yprueba).^2))/length(Yprueba);
    
    Yestimado(Yestimado>=0)=1;
    Yestimado(Yestimado<0)=0;
    
    MatrizConfusion = zeros(NumClases,NumClases);
    for a=1:size(Xprueba,1)
    % Se le suma  1 a Yprueba para llevar las clases de los valores
    % 0 y 1 a 1 y 2, para que cuadren los indices
    MatrizConfusion(Yestimado(a)+1,Yprueba(a)+1) = MatrizConfusion(Yestimado(a)+1,Yprueba(a)+1) + 1;
    end        
    EficienciaTest(i) = sum(diag(MatrizConfusion))/sum(sum(MatrizConfusion));
    % Cero el Ojo Abierto (Positivo) y Uno el Ojo Cerrado (Negativo)
    SensibilidadTest(i) = MatrizConfusion(1,1) / sum(MatrizConfusion(1,:));
    EspecificidadTest(i) = MatrizConfusion(2,2) / sum(MatrizConfusion(2,:));

  end  
    Texto=['------Resultado Regresión Logística grado: ',num2str(grado),' ---------'];
    disp(Texto)
    %Texto=['Es error cuadrático medio es: ',num2str(ECM)];
    %disp(Texto)
    %Texto=strcat('La eficiencia en prueba es: ',{' '},num2str(Eficiencia*100),'%');
    Texto=['La eficiencia en prueba es: ',num2str(mean(EficienciaTest)),' +- ',num2str(std(EficienciaTest))];
    disp(Texto);
    Texto=['La sensibilidad en prueba es: ',num2str(mean(SensibilidadTest)),' +- ',num2str(std(SensibilidadTest))];
    disp(Texto);
    Texto=['La especificidad en prueba es: ',num2str(mean(EspecificidadTest)),' +- ',num2str(std(EspecificidadTest))];
    disp(Texto);
    
elseif seleccion == 2 %SVM
    
    fold=10;
    EficienciaTest=zeros(1,fold);
    SensibilidadTest = zeros(1,fold);
    EspecificidadTest = zeros(1,fold);
    
    %--------------------Selección del grado del polinomio
    
    boxConstraint=input('\nIngrese el boxConstraint para la SVM:\n----> ');
    vkernel=input('\nIngrese el tipo de kernel (1:gauss o 2:linear):\n----> ');   
    if vkernel == 1
       kernel = 'gauss'; 
       gamma = input('\nIngrese el valor de gamma para el kernel gaussiano):\n----> ');
    else
        kernel = 'linear';
        gamma = 0;
    end 
        
  for i=1:fold  
    %--------------------Divisón de las muestras        
    %{
    Xentrenamiento: muestras para entrenar el sistema
    Yentrenamiento: salida del sistema para entrenar
    Xprueba: muestra para probar el sistema
    Yentrenamiento: salida del sistema para validar
     
    %}
    %rng('default');%Semilla aleatoria
    ind=randperm(filas); %%% Se seleccionan los indices de forma aleatoria
    tope=filas*0.9;%Se entrena con el 90% de las muestras
    Xentrenamiento=X(ind(1:tope),:);
    Xprueba=X(ind(tope+1:end),:);    
    Yentrenamiento=Y(ind(1:tope),:);
    Yprueba=Y(ind(tope+1:end),:);
    
    
    %-----------------Normalización        
    [Xentrenamiento,mu,sigma]=zscore(Xentrenamiento);
    Xprueba=normalizar(Xprueba,mu,sigma);
      
    %----------------Se entrena el modelo
    % Salidas en -1 para la clase 0 y 1 para la clase 1
    Yentrenamiento(Yentrenamiento == 0) = -1;
    Modelo=entrenarSVM(Xentrenamiento,Yentrenamiento, 'c', boxConstraint, gamma,kernel);            
    
    %--------------Se prueba
    Yestimado = testSVM(Modelo,Xprueba);
    
    % Estimados en -1 para la clase 0 y en 1 para la clase 1
    Yestimado(Yestimado == -1) = 0;
    
    MatrizConfusion = zeros(NumClases,NumClases);
    for a=1:size(Xprueba,1)
    % Se le suma  1 a Yprueba para llevar las clases de los valores
    % 0 y 1 a 1 y 2, para que cuadren los indices
    MatrizConfusion(Yestimado(a)+1,Yprueba(a)+1) = MatrizConfusion(Yestimado(a)+1,Yprueba(a)+1) + 1;
    end        
    EficienciaTest(i) = sum(diag(MatrizConfusion))/sum(sum(MatrizConfusion));
    % Cero el Ojo Abierto (Positivo) y Uno el Ojo Cerrado (Negativo)
    SensibilidadTest(i) = MatrizConfusion(1,1) / sum(MatrizConfusion(1,:));
    EspecificidadTest(i) = MatrizConfusion(2,2) / sum(MatrizConfusion(2,:));

  end  
    Texto=['------Resultado Regresión Logística grado: ',num2str(grado),' ---------'];
    disp(Texto)
    %Texto=['Es error cuadrático medio es: ',num2str(ECM)];
    %disp(Texto)
    %Texto=strcat('La eficiencia en prueba es: ',{' '},num2str(Eficiencia*100),'%');
    Texto=['La eficiencia en prueba es: ',num2str(mean(EficienciaTest)),' +- ',num2str(std(EficienciaTest))];
    disp(Texto);
    Texto=['La sensibilidad en prueba es: ',num2str(mean(SensibilidadTest)),' +- ',num2str(std(SensibilidadTest))];
    disp(Texto);
    Texto=['La especificidad en prueba es: ',num2str(mean(EspecificidadTest)),' +- ',num2str(std(EspecificidadTest))];
    disp(Texto);

    
elseif seleccion==3 %Modelo de Mezclas Gaussianas
        Rept=10;
        Mezclas=input('\nIngrese el numero de mezclas del GMM:\n---> ');
        tipoMatriz=input('\nIngrese el tipo de matriz de covarianza:\n(1) Esférica\n(2) Diagonal\n(3) Completa\n---> ');
        if tipoMatriz==1
            tipoMatriz='spherical';
        elseif tipoMatriz==2
            tipoMatriz='diag';
        elseif tipoMatriz==3
            tipoMatriz='full';
        end
     EficienciaTest=zeros(1,Rept);
     SensibilidadTest = zeros(1,Rept);
     EspecificidadTest = zeros(1,Rept);
     for i=1:Rept
        %{
        Xentrenamiento: muestras para entrenar el sistema
        Yentrenamiento: salida del sistema para entrenar
        Xprueba: muestra para probar el sistema
        Yentrenamiento: salida del sistema para validar

        %}
        %rng('default');%Semilla aleatoria
        ind=randperm(filas); %%% Se seleccionan los indices de forma aleatoria
        tope=filas*0.9;%Se entrena con el 90% de las muestras
        Xentrenamiento=X(ind(1:tope),:);
        Xprueba=X(ind(tope+1:end),:);    
        Yentrenamiento=Y(ind(1:tope),:);
        Yprueba=Y(ind(tope+1:end),:);


        %-----------------Normalización        
        [Xentrenamiento,mu,sigma]=zscore(Xentrenamiento);
        Xprueba=normalizar(Xprueba,mu,sigma);
        
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%% Entrenamiento de los modelos%%%
        
        vInd=(Yentrenamiento == 0);
        XtrainC1 = Xentrenamiento(vInd,:);
        if ~isempty(XtrainC1)
            Modelo1=entrenarGMM(XtrainC1,Mezclas,tipoMatriz);
        else
            error('No hay muestras de todas las clases para el entrenamiento');
        end
        
        vInd=(Yentrenamiento == 1);
        XtrainC2 = Xentrenamiento(vInd,:);
        if ~isempty(XtrainC2)
            Modelo2=entrenarGMM(XtrainC2,Mezclas,tipoMatriz);
        else
            error('No hay muestras de todas las clases para el entrenamiento');
        end
        
       
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%% Validación de los modelos. %%%
        
        probClase1=testGMM(Modelo1,Xprueba);
        probClase2=testGMM(Modelo2,Xprueba);
        Matriz=[probClase1,probClase2];
        
        [~,Yestimado] = max(Matriz,[],2);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        MatrizConfusion = zeros(NumClases,NumClases);
        for a=1:size(Xprueba,1)
            % Se le suma  1 a Yprueba para llevar las clases de los valores
            % 0 y 1 a 1 y 2, para que cuadren los indices
            MatrizConfusion(Yestimado(a),Yprueba(a)+1) = MatrizConfusion(Yestimado(a),Yprueba(a)+1) + 1;
        end        
        EficienciaTest(i) = sum(diag(MatrizConfusion))/sum(sum(MatrizConfusion));
        % Cero el Ojo Abierto (Positivo) y Uno el Ojo Cerrado (Negativo)
        SensibilidadTest(i) = MatrizConfusion(1,1) / sum(MatrizConfusion(1,:));
        EspecificidadTest(i) = MatrizConfusion(2,2) / sum(MatrizConfusion(2,:));
     end        
    Texto=['------ Resultado GMM, con Mezclas= ',num2str(Mezclas),' Tipo Matriz= ',num2str(tipoMatriz),'---------'];
    disp(Texto)
    Texto=['La eficiencia en prueba es: ',num2str(mean(EficienciaTest)),' +- ',num2str(std(EficienciaTest))];
    disp(Texto);
    Texto=['La sensibilidad en prueba es: ',num2str(mean(SensibilidadTest)),' +- ',num2str(std(SensibilidadTest))];
    disp(Texto);
    Texto=['La especificidad en prueba es: ',num2str(mean(EspecificidadTest)),' +- ',num2str(std(EspecificidadTest))];
    disp(Texto);
    
elseif seleccion==4 %K vecinos más cercanos
    fold=10;  
    k=input('\nIngrese la cantidad de vecinos :\n----> ');
    EficienciaTest=zeros(1,fold);
    SensibilidadTest = zeros(1,fold);
    EspecificidadTest = zeros(1,fold);
    for i=1:fold

        %{
        Xentrenamiento: muestras para entrenar el sistema
        Yentrenamiento: salida del sistema para entrenar
        Xprueba: muestra para probar el sistema
        Yentrenamiento: salida del sistema para validar

        %}
        %rng('default');%Semilla aleatoria
        ind=randperm(filas); %%% Se seleccionan los indices de forma aleatoria
        tope=filas*0.9;%Se entrena con el 90% de las muestras
        Xentrenamiento=X(ind(1:tope),:);
        Xprueba=X(ind(tope+1:end),:);    
        Yentrenamiento=Y(ind(1:tope),:);
        Yprueba=Y(ind(tope+1:end),:);


        %-----------------Normalización        
        [Xentrenamiento,mu,sigma]=zscore(Xentrenamiento);
        Xprueba=normalizar(Xprueba,mu,sigma);
        
        %Entrenar Modelo K Neighbors
        Yestimado = vecinosCercanos(Xprueba,Xentrenamiento,Yentrenamiento,k);
        
        %------------------Resultado
        MatrizConfusion = zeros(NumClases,NumClases);
        for a=1:size(Xprueba,1)
            % Se le suma  1 a Yprueba para llevar las clases de los valores
            % 0 y 1 a 1 y 2, para que cuadren los indices
            MatrizConfusion(Yestimado(a)+1,Yprueba(a)+1) = MatrizConfusion(Yestimado(a)+1,Yprueba(a)+1) + 1;
        end        
        EficienciaTest(i) = sum(diag(MatrizConfusion))/sum(sum(MatrizConfusion));
        % Cero el Ojo Abierto (Positivo) y Uno el Ojo Cerrado (Negativo)
        SensibilidadTest(i) = MatrizConfusion(1,1) / sum(MatrizConfusion(1,:));
        EspecificidadTest(i) = MatrizConfusion(2,2) / sum(MatrizConfusion(2,:));        
    end
    Texto=['------ Resultado K Neighbors, con K= ',num2str(k),'---------'];
    disp(Texto)
    %Texto=['Es error cuadrático medio es: ',num2str(ECM)];
    %disp(Texto)
    %Texto=strcat('La eficiencia en prueba es: ',{' '},num2str(Eficiencia*100),'%');
    %figure(1);
    %hold on;
    %plot(k,(Eficiencia*100),'.g');
    %hold on;
    %plot(k,Error*100,'.r');
    %title('Eficiencia y Error v.s K');
    %ylabel('%');
    %xlabel('# Vecinos (K)');
    Texto=['La eficiencia en prueba es: ',num2str(mean(EficienciaTest)),' +- ',num2str(std(EficienciaTest))];
    disp(Texto);
    Texto=['La sensibilidad en prueba es: ',num2str(mean(SensibilidadTest)),' +- ',num2str(std(SensibilidadTest))];
    disp(Texto);
    Texto=['La especificidad en prueba es: ',num2str(mean(EspecificidadTest)),' +- ',num2str(std(EspecificidadTest))];
    disp(Texto);
   
   %{
   hold on;
   plot(1:n,vectorEficiencia,'g');
   plot(1:n,vectorError,'r');
   title('Eficiencia y Error v.s K');
   legend('Eficiencia','Error')
   ylabel('%');
   xlabel('# Vecinos (K)');
   %}
elseif seleccion==5 %RNA
    fold=10;
    EficienciaTest=zeros(1,fold);
    SensibilidadTest = zeros(1,fold);
    EspecificidadTest = zeros(1,fold); 
    capas=input('\nIngrese el numero de capas ocultas:\n---> ');
    neuronas=input('\nIngrese el numero de neuronas:\n---> ');
    %-----------------Vector Numero Neuronas
    vectorNeuronas = ones(1,capas);
    vectorNeuronas = neuronas * vectorNeuronas;    
    
    for i=1:fold
    %--------------------Divisón de las muestras        
    %{
    Xentrenamiento: muestras para entrenar el sistema
    Yentrenamiento: salida del sistema para entrenar
    Xprueba: muestra para probar el sistema
    Yentrenamiento: salida del sistema para validar
     
    %}
    %rng('default');%Semilla aleatoria
    ind=randperm(filas); %%% Se seleccionan los indices de forma aleatoria
    tope=filas*0.9;%Se entrena con el 90% de las muestras
    Xentrenamiento=X(ind(1:tope),:);
    Xprueba=X(ind(tope+1:end),:);    
    Yentrenamiento=Y(ind(1:tope),:);
    Yprueba=Y(ind(tope+1:end),:);
    
    
    %-----------------Normalización        
    [Xentrenamiento,mu,sigma]=zscore(Xentrenamiento);
    Xprueba=normalizar(Xprueba,mu,sigma);
    
    %------------------Entrenamiento RNA
    net = patternnet(vectorNeuronas);
    net = train(net,Xentrenamiento',Yentrenamiento');
    Yestimado = net(Xprueba');
    
    Yestimado = round(Yestimado);
    
    %------------------Resultado
    MatrizConfusion = zeros(NumClases,NumClases);
    for a=1:size(Xprueba,1)
		% Se le suma  1 a Yprueba para llevar las clases de los valores
		% 0 y 1 a 1 y 2, para que cuadren los indices
		MatrizConfusion(Yestimado(a)+1,Yprueba(a)+1) = MatrizConfusion(Yestimado(a)+1,Yprueba(a)+1) + 1;
    end        
	EficienciaTest(i) = sum(diag(MatrizConfusion))/sum(sum(MatrizConfusion));
	% Cero el Ojo Abierto (Positivo) y Uno el Ojo Cerrado (Negativo)
	SensibilidadTest(i) = MatrizConfusion(1,1) / sum(MatrizConfusion(1,:));
	EspecificidadTest(i) = MatrizConfusion(2,2) / sum(MatrizConfusion(2,:));
    
    end
     %Texto=['Es error cuadrático medio es: ',num2str(ECM)];
    %disp(Texto)
    %Texto=strcat('La eficiencia en prueba es: ',{' '},num2str(Eficiencia*100),'%');
    Texto=['------ Resultado RNA, con Capas Ocultas= ',num2str(capas),' Neuronas= ',num2str(neuronas),'---------'];
    disp(Texto);
    Texto=['La eficiencia en prueba es: ',num2str(mean(EficienciaTest)),' +- ',num2str(std(EficienciaTest))];
    disp(Texto);
    Texto=['La sensibilidad en prueba es: ',num2str(mean(SensibilidadTest)),' +- ',num2str(std(SensibilidadTest))];
    disp(Texto);
    Texto=['La especificidad en prueba es: ',num2str(mean(EspecificidadTest)),' +- ',num2str(std(EspecificidadTest))];
    disp(Texto);
elseif seleccion == 6 %Random Forest
    fold=10;
    NumTrees=input('\nIngrese el número de árboles para el comité:\n----> ');
    EficienciaTest=zeros(1,fold);
    SensibilidadTest = zeros(1,fold);
    EspecificidadTest = zeros(1,fold);
    for i=1:fold
    %--------------------Divisón de las muestras        
    %{
    Xentrenamiento: muestras para entrenar el sistema
    Yentrenamiento: salida del sistema para entrenar
    Xprueba: muestra para probar el sistema
    Yentrenamiento: salida del sistema para validar
     
    %}
    %rng('default');%Semilla aleatoria
    ind=randperm(filas); %%% Se seleccionan los indices de forma aleatoria
    tope=filas*0.9;%Se entrena con el 90% de las muestras
    Xentrenamiento=X(ind(1:tope),:);
    Xprueba=X(ind(tope+1:end),:);    
    Yentrenamiento=Y(ind(1:tope),:);
    Yprueba=Y(ind(tope+1:end),:);
    
    
    %-----------------Normalización        
    [Xentrenamiento,mu,sigma]=zscore(Xentrenamiento);
    Xprueba=normalizar(Xprueba,mu,sigma);    
    
    %Entrenar Random Forest
    forest =entrenarFOREST(NumTrees,Xentrenamiento,Yentrenamiento);
    
    %------------resultado  
    Yestimado = predict(forest,Xprueba);
    S = sprintf('%s*', Yestimado{:});
    Yestimado = sscanf(S, '%f*');
    MatrizConfusion = zeros(NumClases,NumClases);
    for a=1:size(Xprueba,1)
		% Se le suma  1 a Yprueba para llevar las clases de los valores
		% 0 y 1 a 1 y 2, para que cuadren los indices
		MatrizConfusion(Yestimado(a)+1,Yprueba(a)+1) = MatrizConfusion(Yestimado(a)+1,Yprueba(a)+1) + 1;
    end        
	EficienciaTest(i) = sum(diag(MatrizConfusion))/sum(sum(MatrizConfusion));
	% Cero el Ojo Abierto (Positivo) y Uno el Ojo Cerrado (Negativo)
	SensibilidadTest(i) = MatrizConfusion(1,1) / sum(MatrizConfusion(1,:));
	EspecificidadTest(i) = MatrizConfusion(2,2) / sum(MatrizConfusion(2,:));
    end 
      
    Texto=['------ Resultado Random Forest #trees= ',num2str(NumTrees),' ---------'];
    disp(Texto)
    Texto=['La eficiencia en prueba es: ',num2str(mean(EficienciaTest)),' +- ',num2str(std(EficienciaTest))];
    disp(Texto);
    Texto=['La sensibilidad en prueba es: ',num2str(mean(SensibilidadTest)),' +- ',num2str(std(SensibilidadTest))];
    disp(Texto);
    Texto=['La especificidad en prueba es: ',num2str(mean(EspecificidadTest)),' +- ',num2str(std(EspecificidadTest))];
    disp(Texto);   

    
   %{ 
    hold on;
   plot(1:NumTreesMax,vectorEficiencia,'g');
   plot(1:NumTreesMax,vectorError,'r');
   title('Eficiencia y Error v.s K');
   legend('Eficiencia','Error')
   ylabel('%');
   xlabel('#Árboles en comité');
    %}
elseif seleccion == 7 %Arbol de decisión   
    fold = 10;
    EficienciaTest=zeros(1,fold);
    SensibilidadTest = zeros(1,fold);
    EspecificidadTest = zeros(1,fold);
    nivelPoda=input('Ingrese el nivel de poda:\n--->');
    for i=1:fold
        %{
        Xentrenamiento: muestras para entrenar el sistema
        Yentrenamiento: salida del sistema para entrenar
        Xprueba: muestra para probar el sistema
        Yentrenamiento: salida del sistema para validar

        %}
        %rng('default');%Semilla aleatoria
        ind=randperm(filas); %%% Se seleccionan los indices de forma aleatoria
        tope=filas*0.9;%Se entrena con el 90% de las muestras
        Xentrenamiento=X(ind(1:tope),:);
        Xprueba=X(ind(tope+1:end),:);    
        Yentrenamiento=Y(ind(1:tope),:);
        Yprueba=Y(ind(tope+1:end),:);


        %-----------------Normalización        
        [Xentrenamiento,mu,sigma]=zscore(Xentrenamiento);
        Xprueba=normalizar(Xprueba,mu,sigma);
  
       %-----Entrenamiento TREE
        tree = fitctree(Xentrenamiento,Yentrenamiento);
        treePruned = prune(tree,'level',nivelPoda);
        Yestimado = predict(treePruned,Xprueba);
    
        %------------resultado
        MatrizConfusion = zeros(NumClases,NumClases);
        for a=1:size(Xprueba,1)
            % Se le suma  1 a Yprueba para llevar las clases de los valores
            % 0 y 1 a 1 y 2, para que cuadren los indices
            MatrizConfusion(Yestimado(a)+1,Yprueba(a)+1) = MatrizConfusion(Yestimado(a)+1,Yprueba(a)+1) + 1;
        end        
        EficienciaTest(i) = sum(diag(MatrizConfusion))/sum(sum(MatrizConfusion));
        % Cero el Ojo Abierto (Positivo) y Uno el Ojo Cerrado (Negativo)
        SensibilidadTest(i) = MatrizConfusion(1,1) / sum(MatrizConfusion(1,:));
        EspecificidadTest(i) = MatrizConfusion(2,2) / sum(MatrizConfusion(2,:));
    end  
    Texto=['------ Resultado Árbol de decisión con nivel de poda= ',num2str(nivelPoda),'---------'];
    disp(Texto)
    Texto=['La eficiencia en prueba es: ',num2str(mean(EficienciaTest)),' +- ',num2str(std(EficienciaTest))];
    disp(Texto);
    Texto=['La sensibilidad en prueba es: ',num2str(mean(SensibilidadTest)),' +- ',num2str(std(SensibilidadTest))];
    disp(Texto);
    Texto=['La especificidad en prueba es: ',num2str(mean(EspecificidadTest)),' +- ',num2str(std(EspecificidadTest))];
    disp(Texto);  
    %view(tree,'Mode','graph')
 %{  
   hold on;
   plot(0:n,vectorEficiencia,'g');
   plot(0:n,vectorError,'r');
   title('Eficiencia y Error v.s K');
   legend('Eficiencia','Error')
   ylabel('%');
   xlabel('Nivel de poda');
   view(tree,'Mode','graph')
   view(treePruned,'Mode','graph')
  %}
elseif seleccion==8 %Ventana de parzen
    fold=10;
    h=input('Ingrese la ventana de suavizado:\n---> ');%Ventana de suavizado
    EficienciaTest=zeros(1,fold);
    SensibilidadTest = zeros(1,fold);
    EspecificidadTest = zeros(1,fold);
    for i=1:fold
        %--------------------Divisón de las muestras        
        %{
        Xentrenamiento: muestras para entrenar el sistema
        Yentrenamiento: salida del sistema para entrenar
        Xprueba: muestra para probar el sistema
        Yentrenamiento: salida del sistema para validar
     
        %}
        %rng('default');%Semilla aleatoria
        ind=randperm(filas); %%% Se seleccionan los indices de forma aleatoria
        tope=filas*0.9;%Se entrena con el 90% de las muestras
        Xentrenamiento=X(ind(1:tope),:);
        Xprueba=X(ind(tope+1:end),:);    
        Yentrenamiento=Y(ind(1:tope),:);
        Yprueba=Y(ind(tope+1:end),:);
    
    
        %-----------------Normalización        
        [Xentrenamiento,mu,sigma]=zscore(Xentrenamiento);
        Xprueba=normalizar(Xprueba,mu,sigma);    
    
        %-----------------Entrenamiento Ventana de Parzen
        Yestimado = ventanaParzen(Xprueba,Xentrenamiento,Yentrenamiento,h,'class');
        %La salida es 1 o 2
    
        %-----------------Resultado
        MatrizConfusion = zeros(NumClases,NumClases);
        for a=1:size(Xprueba,1)
            % Se le suma  1 a Yprueba para llevar las clases de los valores
            % 0 y 1 a 1 y 2, para que cuadren los indices
		MatrizConfusion(Yestimado(a),Yprueba(a)+1) = MatrizConfusion(Yestimado(a),Yprueba(a)+1) + 1;
        end        
        EficienciaTest(i) = sum(diag(MatrizConfusion))/sum(sum(MatrizConfusion));
        % Cero el Ojo Abierto (Positivo) y Uno el Ojo Cerrado (Negativo)
        SensibilidadTest(i) = MatrizConfusion(1,1) / sum(MatrizConfusion(1,:));
        EspecificidadTest(i) = MatrizConfusion(2,2) / sum(MatrizConfusion(2,:));
    end    

    Texto=['------ Resultado ventana de Parzen con h= ',num2str(h),' ---------'];
    disp(Texto)
    Texto=['La eficiencia en prueba es: ',num2str(mean(EficienciaTest)),' +- ',num2str(std(EficienciaTest))];
    disp(Texto);
    Texto=['La sensibilidad en prueba es: ',num2str(mean(SensibilidadTest)),' +- ',num2str(std(SensibilidadTest))];
    disp(Texto);
    Texto=['La especificidad en prueba es: ',num2str(mean(EspecificidadTest)),' +- ',num2str(std(EspecificidadTest))];
    disp(Texto); 
    
 %{
    hold on;
   plot((1:10)/100,vectorEficiencia,'g');
   plot((1:10)/100,vectorError,'r');
   title('Eficiencia y Error v.s h');
   legend('Eficiencia','Error')
   ylabel('%');
   xlabel('Ventana de suavizado (h)');
 %}
 elseif seleccion == 9 %Reducción de dimensión
    clc;
    seleccion = input('Ingrese:\n(1) Coeficiente de correlación de Pearson\n(2) Cociente discriminante de Fisher\n(3) Selección de características \n(4) PCA\n----> ');
    clc;
    if seleccion == 1%Coeficiente de correlación de Pearson
        disp('***Coeficiente de correlación de Pearson***');
        MatCorr= abs(corrcoef(X));
        lista=zeros(columnas,1);
        for i=1:columnas
         for j=i:columnas
             if MatCorr(i,j)>=0.9 && i~=j
                 lista(j)=1;
                 Texto=['La Característica #: ',num2str(i),' explica la característica #: ',num2str(j),' un ',num2str(MatCorr(i,j)^2),'%'];
                 disp(Texto);
              end
         end
        end
        input('\n\n(Presione Enter)');
        clc;
        disp('***Coeficiente de correlación de Pearson***');
        disp('***Características candidatas a ser eliminadas***');
        for i=1:columnas
            if lista(i)==1            
                Texto=['Característica #: ',num2str(i)];
                disp(Texto);
            end
        end
    elseif seleccion == 2%Cociente discriminante de Fisher
        disp('***Cociente discriminante de Fisher***');
        F=zeros(columnas,1);
        for i=1:columnas
            mediaI=mean(X(:,i));
            varianzaI = var(X(:,i));
            for j=1:columnas
                if i~=j
                    mediaJ=mean(X(:,j));
                    varianzaJ = var(X(:,j));
                    F(i)= F(i)+(((mediaI - mediaJ)^2)/ (varianzaI + varianzaJ));
                end
            end
        end
        disp('***Sin normalizar***');
        F
        input('\n\n(Presione Enter)');
        clc;
        maxF= max(F);
        F=F./maxF;
        disp('***Cociente discriminante de Fisher***');
        disp('***Normalizado***');
        F
        input('\n\n(Presione Enter)');
        clc;
        disp('***Cociente discriminante de Fisher***');
        disp('***Características candidatas a ser eliminadas***');
        for i=1:columnas
            if F(i) <= 0.5
                Texto=['La Característica #: ',num2str(i),' no presenta una capacidad discriminante'];
                disp(Texto);
            else
                Texto=['La Característica #: ',num2str(i),' permanece'];
                disp(Texto);
            end
        end
        
    elseif seleccion == 3%Selección de características: método wrapper
        fsAcumulado= zeros(columnas,1);
        iteraciones=2;
        for i=1:iteraciones
            clc;
            Texto='[';
            for j=1:i
                Texto=strcat(Texto,'*');
            end
            for j=i:iteraciones
                Texto=strcat(Texto,'_');
            end
            Texto=strcat(Texto,']');
            disp(Texto);
            fs = sequentialfs(@miCriterio,X,Y);
            fsAcumulado = fsAcumulado + fs';
        end
        disp('***Selección de características***');
        disp('***Wrapper: % de importancia de cada característica***');
        fsAcumulado=fsAcumulado./iteraciones;
        fsAcumulado=fsAcumulado.*100;
        fsAcumulado
        input('\n\n(Presione Enter)');
        clc;
        disp('***Selección de características***');
        disp('***Características candidatas a ser eliminadas***');
        topeMinimo=30;
        for i=1:columnas
            if fsAcumulado(i)<topeMinimo            
                Texto=['Característica #: ',num2str(i)];
                disp(Texto);
            end
        end
    elseif seleccion == 4   
        %PCA
        fold = 10;
        EficienciaTestRF=zeros(1,fold);
        SensibilidadTestRF = zeros(1,fold);
        EspecificidadTestRF = zeros(1,fold);
        EficienciaTestParzen=zeros(1,fold);
        SensibilidadTestParzen = zeros(1,fold);
        EspecificidadTestParzen = zeros(1,fold);
        EficienciaTestKVecinos=zeros(1,fold);
        SensibilidadTestKVecinos = zeros(1,fold);
        EspecificidadTestKVecinos = zeros(1,fold);
        for i=1:fold
            %--------------------Divisón de las muestras        
            %{
            Xentrenamiento: muestras para entrenar el sistema
            Yentrenamiento: salida del sistema para entrenar
            Xprueba: muestra para probar el sistema
            Yentrenamiento: salida del sistema para validar
     
            %}
            %rng('default');%Semilla aleatoria
            ind=randperm(filas); %%% Se seleccionan los indices de forma aleatoria
            tope=filas*0.9;%Se entrena con el 90% de las muestras
            Xentrenamiento=X(ind(1:tope),:);
            Xprueba=X(ind(tope+1:end),:);    
            Yentrenamiento=Y(ind(1:tope),:);
            Yprueba=Y(ind(tope+1:end),:);
    
    
            %-----------------Normalización        
            [Xentrenamiento,mu,sigma]=zscore(Xentrenamiento);
            Xprueba=normalizar(Xprueba,mu,sigma);    
    
            %Se aplica PCA
            [co, score, lat] = pca(Xentrenamiento);
            L = cumsum(lat);
            L = L/L(end);
            p = find(L >= 0.9);
            p = p(1);
            w = co(:,1:p);
            Xentrenamiento2 = score(:,1:p);
            Xprueba2 = Xprueba * w;
            
            % Se vuelven a validar los tres mejores modelos.
            
            % 1) Random Forest con 30 árboles
            %Entrenar Random Forest
            forest =entrenarFOREST(30,Xentrenamiento2,Yentrenamiento);
            %------------resultado  
            Yestimado = predict(forest,Xprueba2);
            S = sprintf('%s*', Yestimado{:});
            Yestimado = sscanf(S, '%f*');
            
            MatrizConfusion = zeros(NumClases,NumClases);
            for a=1:size(Xprueba2,1)
                % Se le suma  1 a Yprueba para llevar las clases de los valores
                % 0 y 1 a 1 y 2, para que cuadren los indices
                MatrizConfusion(Yestimado(a)+1,Yprueba(a)+1) = MatrizConfusion(Yestimado(a)+1,Yprueba(a)+1) + 1;
            end        
            EficienciaTestRF(i) = sum(diag(MatrizConfusion))/sum(sum(MatrizConfusion));
            % Cero el Ojo Abierto (Positivo) y Uno el Ojo Cerrado (Negativo)
            SensibilidadTestRF(i) = MatrizConfusion(1,1) / sum(MatrizConfusion(1,:));
            EspecificidadTestRF(i) = MatrizConfusion(2,2) / sum(MatrizConfusion(2,:));
            
            % 2) Ventana de Parzen con h = 0,05
            %-----------------Entrenamiento Ventana de Parzen
            Yestimado = ventanaParzen(Xprueba2,Xentrenamiento2,Yentrenamiento,0.05,'class');
            %La salida es 1 o 2
    
            %-----------------Resultado
            MatrizConfusion = zeros(NumClases,NumClases);
            for a=1:size(Xprueba2,1)
                % Se le suma  1 a Yprueba para llevar las clases de los valores
                % 0 y 1 a 1 y 2, para que cuadren los indices
                MatrizConfusion(Yestimado(a),Yprueba(a)+1) = MatrizConfusion(Yestimado(a),Yprueba(a)+1) + 1;
            end        
            EficienciaTestParzen(i) = sum(diag(MatrizConfusion))/sum(sum(MatrizConfusion));
            % Cero el Ojo Abierto (Positivo) y Uno el Ojo Cerrado (Negativo)
            SensibilidadTestParzen(i) = MatrizConfusion(1,1) / sum(MatrizConfusion(1,:));
            EspecificidadTestParzen(i) = MatrizConfusion(2,2) / sum(MatrizConfusion(2,:));
            
            % 3) K vecinos con 6 vecinos
            %Entrenar Modelo K Neighbors
            Yestimado = vecinosCercanos(Xprueba2,Xentrenamiento2,Yentrenamiento,6);

            %------------------Resultado
            MatrizConfusion = zeros(NumClases,NumClases);
            for a=1:size(Xprueba2,1)
                % Se le suma  1 a Yprueba para llevar las clases de los valores
                % 0 y 1 a 1 y 2, para que cuadren los indices
                MatrizConfusion(Yestimado(a)+1,Yprueba(a)+1) = MatrizConfusion(Yestimado(a)+1,Yprueba(a)+1) + 1;
            end        
            EficienciaTestKVecinos(i) = sum(diag(MatrizConfusion))/sum(sum(MatrizConfusion));
            % Cero el Ojo Abierto (Positivo) y Uno el Ojo Cerrado (Negativo)
            SensibilidadTestKVecinos(i) = MatrizConfusion(1,1) / sum(MatrizConfusion(1,:));
            EspecificidadTestKVecinos(i) = MatrizConfusion(2,2) / sum(MatrizConfusion(2,:));
            
        end    
        Texto=['------ Resultado Random Forest luego de aplicar pca #trees= ',num2str(30),' ---------'];
        disp(Texto)
        Texto=['La eficiencia en prueba es: ',num2str(mean(EficienciaTestRF)),' +- ',num2str(std(EficienciaTestRF))];
        disp(Texto);
        Texto=['La sensibilidad en prueba es: ',num2str(mean(SensibilidadTestRF)),' +- ',num2str(std(SensibilidadTestRF))];
        disp(Texto);
        Texto=['La especificidad en prueba es: ',num2str(mean(EspecificidadTestRF)),' +- ',num2str(std(EspecificidadTestRF))];
        disp(Texto);
        
        Texto=['------ Resultado ventana de Parzen luego de aplicar pca con h= ',num2str(0.05),' ---------'];
        disp(Texto)
        Texto=['La eficiencia en prueba es: ',num2str(mean(EficienciaTestParzen)),' +- ',num2str(std(EficienciaTestParzen))];
        disp(Texto);
        Texto=['La sensibilidad en prueba es: ',num2str(mean(SensibilidadTestParzen)),' +- ',num2str(std(SensibilidadTestParzen))];
        disp(Texto);
        Texto=['La especificidad en prueba es: ',num2str(mean(EspecificidadTestParzen)),' +- ',num2str(std(EspecificidadTestParzen))];
        disp(Texto);
        
        Texto=['------ Resultado K Neighbors luego de aplicar pca con h, con K= ',num2str(6),'---------'];
        disp(Texto);
        Texto=['La eficiencia en prueba es: ',num2str(mean(EficienciaTestKVecinos)),' +- ',num2str(std(EficienciaTestKVecinos))];
        disp(Texto);
        Texto=['La sensibilidad en prueba es: ',num2str(mean(SensibilidadTestKVecinos)),' +- ',num2str(std(SensibilidadTestKVecinos))];
        disp(Texto);
        Texto=['La especificidad en prueba es: ',num2str(mean(EspecificidadTestKVecinos)),' +- ',num2str(std(EspecificidadTestKVecinos))];
        disp(Texto);
    end
 elseif seleccion == 10 
     % Normalización   
    [X,mu,sigma]=zscore(X);
    Xtest=normalizar(Xtest,mu,sigma);    
    
     %Predicción Random Forest
    NumTrees=30;      
       
    %Entrenar Random Forest
    forest =entrenarFOREST(NumTrees,X,Y);
    
    %------------resultado  
    YtestRandomForest = predict(forest,Xtest);
    S = sprintf('%s*', YtestRandomForest{:});
    YtestRandomForest = sscanf(S, '%f*');

 %Predición Ventana de parzen
    h = 0.05;
    YtestVentanaParzen = ventanaParzen(Xtest,X,Y,h,'class');
    %La salida es 1 o 2
    %Se resta 1, para que quede 0 o 1
    YtestVentanaParzen=YtestVentanaParzen-1.;
     
  % Se guardan las variables
  save('Prediccion.mat','YtestRandomForest','YtestVentanaParzen')
  sprintf('Se guardó las predicciones en Prediccion.mat...')  
end
sprintf('\t...Fin de programa...')
