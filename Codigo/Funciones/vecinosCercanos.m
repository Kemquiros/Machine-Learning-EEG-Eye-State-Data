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

function Yesti = vecinosCercanos(Xval,Xent,Yent,k)

    %%% El parametro 'tipo' es el tipo de problema que se va a resolver
    
    %%% La funci贸n debe retornar el valor de predicci贸n Yesti para cada una de 
    %%% las muestras en Xval. Por esa raz贸n Yesti se inicializa como un vectores 
    %%% de ceros, de dimensi贸n M.

    %Cantidad de datos entrenamiento
    N=size(Xent,1);
    %Cantidad de datos validaci髇
    M=size(Xval,1);
    indices=zeros(N,1);
    %Vector columna M validacion
    Yesti=zeros(M,1);
    %Vector columna N entrenamiento
    %dis=zeros(N,1);
    
    %Grafica los datos de entrada
    %{
    for i=1:N
        figure(3);
        hold on;
        if(Yent(i)==0)
            plot(Xent(i,1),Xent(i,2),'xr');
        end
        if(Yent(i)==1)
            plot(Xent(i,1),Xent(i,2),'xb');
        end            
    end
    %}
     
       for j=1:M
           
            %dis = distanciaEuclidiana(Xent, Xval(j));
            vectorResta=zeros(N,1);
            for i=1:N
                vectorResta(i)=norm(Xval(j,:)-Xent(i,:));
            end
            [vectorResta,indices]=sort(vectorResta);
            
            clase1=0;
            clase2=0;
            
            for baby=1:k
              if( Yent(indices(baby)) ==0 )
                  clase1=clase1+1;
              end
              if( Yent(indices(baby)) ==1 )
                  clase2=clase2+1;
              end
            end
            if(clase1>=clase2)
                
                    Yesti(j)=0;
                    %{
                    figure(3);
                    hold on;
                    plot(Xval(j,1),Xval(j,2));
                    pause(0.5);
                    plot(Xval(j,1),Xval(j,2),'xr');
                    pause(0.2);
                    %}
                else
                    Yesti(j)=1;
                    %{
                    figure(3);
                    hold on;
                    plot(Xval(j,1),Xval(j,2));
                    pause(0.5);
                    plot(Xval(j,1),Xval(j,2),'xg');
                    pause(0.2);
                    %}
            end
            
                                      
        end
         

end
