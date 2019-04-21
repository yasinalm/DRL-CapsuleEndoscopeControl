function BB = BB_Packing(ri,rj)
n_s = size(ri,2);
n_a = size(rj,2);

BB = zeros(3*n_s,3*n_a);
mu = 4*pi*1e-7;

for i_s=1:n_s    % Row: sensor index
    
    r_s = ri(:,i_s);
        
    for i_a = 1:n_a % column : actuator index
        
        r_a = rj(:,i_a);
    
        p = r_s-r_a;
        
        x = p(1);
        y = p(2);
        z = p(3);
        
        r = sqrt(x^2+y^2+z^2);
        x = x/r;
        y = y/r;
        z = z/r;
        
        P = [3*x^2 - 1,     3*x*y,     3*x*z;
            3*x*y, 3*y^2 - 1,     3*y*z;
            3*x*z,     3*y*z, 3*z^2 - 1];
        
        
        P = P/r^3;
        
        BB(3*i_s-2:3*i_s,3*i_a-2:3*i_a) =  P;
        
        
    end
end
BB = mu/4/pi *BB;

end