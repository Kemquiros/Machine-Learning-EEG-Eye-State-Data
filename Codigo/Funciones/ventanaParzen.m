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

  function Yesti = ventanaParzen(Xval,Xent,Yent,h,tipo)

          %%% La función debe retornar el valor de predicción Yesti para cada una de 
	  %%% las muestras en Xval. Por esa razón Yesti se inicializa como un vectores 
	  %%% de ceros, de dimensión M.
  
      %Validacion
      M=size(Xval,1);
      %Entrenamiento
      N=size(Xent,1);
      
      Yesti=zeros(M,1);

      if strcmp(tipo,'regress')
      
      for j=1:M
	    %%% Complete el codigo %%%
        algo=0;
      arr = 0;	  
      aba = 0;
        for i=1:N
            algo=gaussianKernel(norm(Xval(j,:)-Xent(i,:))/h);
            arr = arr + (algo * Yent(i));
            aba = aba + algo;
        end
            Yesti(j)= arr / aba;
	    %%%%%%%%%%%%%%%%%%%%%%%%%%
	    
	  end
      
      elseif strcmp(tipo,'class')
	  
	  for j=1:M
	      %%% Complete el codigo %%%
            
            sumatoria=zeros(2,1);
            funcion=zeros(2,1);
            n=zeros(2,1);
        for i=1:N
            if(Yent(i)==0)
                %Clase 1
                sumatoria(1)=sumatoria(1) + gaussianKernel(norm(Xval(j,:)-Xent(i,:))/h);
                n(1)=n(1)+1;
            
            elseif(Yent(i)==1)
                %Clase 2
                sumatoria(2)=sumatoria(2) + gaussianKernel(norm(Xval(j,:)-Xent(i,:))/h);
                n(2)=n(2)+1;
            end    
              
        end
        for l=1:2
            funcion(l)=sumatoria(l)/n(l);
        end
        %fprintf('f1 = %f\n',funcion(1));
        %fprintf('f2 = %f\n',funcion(2));       
        [a,Yesti(j)] = max(funcion);
        %fprintf('Y = %d\n',Yesti(j));
        
        
   end
          
          
	      %%%%%%%%%%%%%%%%%%%%%%%%%%
	      
      end
      
	  
      end


