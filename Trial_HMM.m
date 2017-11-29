numstates = 24;
numobservations = 157;

% trans = [0.75,0.25,0,0;
%         0,0.75,0.25,0;
%         0,0,0.75,0.25;
%         0.25,0,0,0.75];
trans = csvread('Transmission_Matrix_Matlab.csv');

emis = rand(numstates,numobservations);
S = sum(emis);
for i=1:numstates
    for j=1:numobservations
        emis(i,j)= emis(i,j)/S(j);
    end
end
init = rand(1,numstates);
M = csvread('Observations.csv');

for k=1:10000
    k
    Observation = round(M(k,:)*100);
    transOld = trans;
    emisOld = emis;
    [trans,emis] = hmmtrain(Observation,trans,emis,'Tolerance',1e-05);
     emis = emis+0.0000000001;
    sum(sum(abs(trans-transOld)))
    sum(sum(abs(emis-emisOld)))
    if (rem(k,50)==0)
        csvwrite('Transmission_Matrix_Matlab_24.csv',trans)
        csvwrite('Emission_Matrix_Matlab_24.csv',emis)
    end
    
end