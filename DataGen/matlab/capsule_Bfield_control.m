rj = load('actuatorPosition.txt');
MM = load('MM.txt');

r_capsule = [0.01;0.01;0.15];   % capsule's position
%n_capsule = [0;1;0];            % desired orientation using unit vector
%n_capsule = n_capsule / norm(n_capsule);

%b_des = n_capsule * 3e-3; % 3 mT along the desired orientation at the capsule's position

BM = BB_Packing(r_capsule,rj)*MM; % B-field actuation matrix
II=[2 2 2 2 2 2 2 2 2]; %set the current values
b_des=II*BM';
n_capsule=b_des/3e-3; %achieved orientation

%II = pinv(BM)*b_des  % current desired