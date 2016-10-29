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

function W = regresionLogistica(X,Y,eta)

[N,D]=size(X);
W = zeros(D,1);
W=W';
for iter = 1:1000
   
    W=W-eta*(1/N)*((sigmoide(W*X'))'-Y)'*X;
    %{
	x=linspace(-11,11,500);
	figure(1);
	k=-W(1)/2;
	h=-W(3)/2;
	r2=-W(5)+h^2+k^2;
	plot(x,-sqrt(r2-(x-h).^2)+k,'.g');
	hold on;
	plot(x,sqrt(r2-(x-h).^2)+k,'.g');
	pause(0.0000001);
    %}
end

end
