function [xcorr,k] = jacobi(A,b,tau,xcorr,Kmax)
    for k=1:Kmax
        r = b - A*xcorr;
        ciao
    end

end