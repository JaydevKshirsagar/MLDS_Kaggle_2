M = csvread('Observations.csv');

TRANS = csvread('Transmission_Matrix_Matlab_24.csv');
EMIS = csvread('Emission_Matrix_Matlab_24.csv');

f = fopen('STATES_Matlab.csv', 'a+');
for k=1:10000
    k
    Observation = round(M(k,:)*100);
    STATES = hmmviterbi(Observation,TRANS,EMIS);
    for i=1:1000
        if i<1000
            fprintf(f,'%f,', STATES(i));
        else
            fprintf(f,'%f\n', STATES(i));
        end
    end
end
fclose(f) ;