% This matlab script is just an inline copy-paste of 
% sparsity_8_sparsespikes_measures 
% Author: Gabriel Peyré
% https://nbviewer.jupyter.org/github/gpeyre/numerical-tours/blob/master/matlab/sparsity_8_sparsespikes_measures.ipynb

addpath('toolbox_signal')
addpath('toolbox_general')

% linewidth for plotting
ms = 20;
lw = 1;

% Sampling grid for the display of functions.
P = 2048*8;
options.P = P;
u = (0:P-1)'/P;

% Set the cutoff pulsation $f_c$ and number of measurements $N=2f_c+1$.
fc = 6;
N = 2*fc+1;

% Fourier transform operator.
Fourier = @(fc,x)exp(-2i*pi*(-fc:fc)'*x(:)');

% Operators $\Phi$ and $\Phi^*$. Note that we assume here implicitely that we use real measures.
Phi  = @(fc,x,a)Fourier(fc,x)*a;
PhiS = @(fc,u,p)real( Fourier(fc,u)'* p );

% Set the spacing $\de$ between the Diracs.
delta = .7/fc;

% Position $x_0$ and amplitude $a_0$ of the input measure $\mu_0=\mu_{x_0,a_0}$ to recover.
x0 = [.5-delta .5 .5+delta]';
a0 = [1 1 -1]';
n = length(x0);

% Position $x_0$ and amplitude $a_0$ of the input measure $\mu_0=\mu_{x_0,a_0}$ to recover.
y0 = Phi(fc,x0,a0);

% Add some noise to obtain $y = y_0 + w$.
% We make sure that the noise has hermitian symmetry to corresponds to the
% Fourier coefficients of a real measure.
sigma = .12 * norm(y0);
w = fftshift( fft(randn(N,1)) ); 
w = w/norm(w) * sigma;
y = y0 + w;

% Display the observed data on the continuous grid $\Phi y_0$.
f0 = PhiS(fc,u,y0);
f = PhiS(fc,u,y);
clf; hold on;
plot(u, [f0 f], 'LineWidth', lw);
stem(x0, 10*sign(a0), 'k.--', 'MarkerSize', ms, 'LineWidth', 1);
axis tight; box on;
legend('\Phi^* y_0', '\Phi^* y');
pbaspect([2 1 1])

set(gcf, 'PaperPosition', [0 0 10 5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [10 5]); %Set the paper to have width 5 and height 5.
saveas(gcf, 'fig/acquis_sdp.pdf');
% system('pdfcrop acquis_sdp.pdf acquis_sdp.pdf');

%% Douglas-Rachford

lambda = 1;

dotp = @(x,y)real(x'*y);
Xmat = @(p,Q)[Q, p; p', 1];
Qmat = @(X)X(1:end-1,1:end-1);
pVec = @(X)X(1:end-1,end);

f = @(p)1/2*norm( y/lambda-p )^2;

Proxf = @(p,gamma)( p + gamma*y/lambda )/(1+gamma);

ProxG = @(X,gamma)perform_sdp_projection(X);

ProxF = @(X,gamma)Xmat( Proxf(pVec(X),gamma/2), perform_sos_projection(Qmat(X)) );

rProxF = @(x,tau)2*ProxF(x,tau)-x;
rProxG = @(x,tau)2*ProxG(x,tau)-x;

X = zeros(2*fc+2);

gamma = 1/10;
mu = 1;

Y = X;
ObjVal = []; ConstrSDP = []; ConstrSOS = [];
niter = 300;
for i=1:niter 
    % record energies
    ObjVal(i) = f(pVec(X));
    ConstrSDP(i) = min(real(eig(X)));
    ConstrSOS(i) = norm(perform_sos_projection(Qmat(X))-Qmat(X), 'fro'); 
    % iterate
	Y = (1-mu/2)*Y + mu/2*rProxF( rProxG(Y,gamma),gamma );        
	X = ProxG(Y,gamma);
end
p = pVec(X);

% Certificate
etaLambda = PhiS(fc,u,p);

%% Root Finding

figure;
stem(x0, sign(abs(a0)), 'k.--', 'MarkerSize', ms, 'LineWidth', lw);
hold on;
plot([0 1],  [1 1], 'k--', 'LineWidth', lw);
hold on;
plot(u, 1-abs(etaLambda).^2, 'b', 'LineWidth', lw);
axis([0 1 -.1 1.1]);
set(gca, 'XTick', [], 'YTick', [0 1]);
box on;
set(gcf, 'PaperPosition', [0 0 10 5]);
set(gcf, 'PaperSize', [10 5]);
saveas(gcf, 'fig/certificate_sdp.pdf');

% Compute the coefficients $c$ of the squared polynomial
c = -conv(p,flipud(conj(p)));
c(N)=1+c(N);

% Compute the roots $R$ of $P$.
R = roots(flipud(c));

% Display the localization of the roots of $P(z)$. Note that roots come in 
% pairs of roots having the same argument.
figure;
plot(real(R),imag(R),'*');
hold on;
plot(cos(2*pi*x0), sin(2*pi*x0),'ro');
plot( exp(1i*linspace(0,2*pi,200)), '--' );
hold off;
legend('Roots','Support of x'); 
axis equal; axis([-1 1 -1 1]*1.5);
set(gcf, 'PaperPosition', [0 0 10 10]);
set(gcf, 'PaperSize', [10 10]);
saveas(gcf, 'fig/roots_sdp.pdf');


tol = 1e-2;
R0 = R(abs(1-abs(R)) < tol);

[~,I]=sort(angle(R0));
R0 = R0(I); R0 = R0(1:2:end);

x = angle(R0)/(2*pi);
x = sort(mod(x,1));

Phix = Fourier(fc,x);
s = sign(real(Phix'*p));

a = real(Phix\y - lambda*pinv(Phix'*Phix)*s );

figure;
stem(x0, a0, 'k.--', 'MarkerSize', ms, 'LineWidth', 1);
hold on;
stem(x, a, 'r.--', 'MarkerSize', ms, 'LineWidth', 1);
axis([0 1 -1.1 1.1]); box on;
legend('Original', 'Recovered');
set(gcf, 'PaperPosition', [0 0 10 5]);
set(gcf, 'PaperSize', [10 5]);
saveas(gcf, 'fig/reconstruction_sdp.pdf');

