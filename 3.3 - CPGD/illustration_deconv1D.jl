using PyPlot, ProgressMeter
using Random, LinearAlgebra

nf = 6 # number of frequency components
phi(x) = real(sum(exp.(im*k*x) for k in -nf:nf))
phi_der(x) = real(sum(im*k*exp.(im*k*x) for k in -nf:nf))

function PGD_1D(r_init, θ_init, r_obs, θ_obs, lambda, alpha, beta, niter; retraction =0)
    ss(r) = abs(r)*r # signed square function
    f_obs(x) = sum(ss.(r_obs) .* phi.(x .- θ_obs)) # observed signal 
    m = length(r_init)
    rs = zeros(m,niter)
    θs = zeros(m,niter)
    gradr, gradθ = zeros(m), zeros(m)
    r, θ = r_init, θ_init
    loss = zeros(niter)
    for iter = 1:niter
        rs[:,iter] = r
        θs[:,iter] = θ
        
        Kxx = phi.(θ .- θ')
        Kyy = phi.(θ_obs .- θ_obs')
        Kxy = phi.(θ .- θ_obs')
        a = ss.(r)/m
        b = ss.(r_obs)
        loss[iter] = (a' * Kxx * a + b' * Kyy * b - 2a' * Kxy * b)/2 + lambda * sum(abs.(a))
        
        for i = 1:m  # gradient computation
            gradr[i] = sign.(r[i]) * (sum(ss.(r) .* phi.(θ[i] .- θ))/m .- sum(ss.(r_obs) .* phi.(θ[i] .- θ_obs))) + lambda
            gradθ[i] = sign.(r[i]) * (sum(ss.(r) .* phi_der.(θ[i] .- θ))/m .- sum(ss.(r_obs) .* phi_der(θ[i].-θ_obs)))       
        end
        if retraction == 0
            r = r .* exp.(- 2 * alpha * gradr) # mirror retraction
        else
            r = r .* (1 .- 2 * alpha * gradr) # canonical retraction
        end
        θ = θ .- beta * gradθ
    end
    return rs, θs, loss
end
   

m0 = 3 # number of spikes ground truth

delta_sep = 0.7 / nf

θ0 = 2π * [0.5-delta_sep; 0.5; 0.5+delta_sep]
w0 = [1; 1; -1]
  
alpha, beta = 0.001, 0.005
lambda =  0.8
r_obs = sign.(w0) .* sqrt.(abs.(w0))
θ_obs = θ0
m = 100
r_init = 0.01*ones(m)
r_init[isodd.(1:m)] *= -1.0
θ_init = range(0,2π*(1-1/m),length=m)
niter = 3000
rs, θs, loss = PGD_1D(r_init, θ_init, r_obs, θ_obs, lambda, alpha, beta, niter, retraction=0)


xs = range(0, 2π, length=2048*8)

# Figure HIERHER to enhance
figure(figsize=[6,3])
plot(θ0, w0, "xC2", label=L"m_{a_0,x_0}")

plot(θs', rs', "k", linewidth=0.8)
# plot((θs .+ 2π)',rs', "k", linewidth=0.8)
# plot((θs .- 2π)',rs',"k", linewidth=0.8)
II = rs[:,end].>=0
I = rs[:,end].<0
plot(θs[II,end],rs[II,end],".C3",markersize=8, label="limit of the positive flow")
plot(θs[II,end] .+ 2π,rs[II,end],".C3",markersize=8)
plot(θs[II,end] .- 2π,rs[II,end],".C3",markersize=8)
plot(θs[I,end],rs[I,end],".C0",markersize=8, label="limit of the negative flow")
plot(θs[I,end].+ 2π,rs[I,end],".C0",markersize=8)
plot(θs[I,end].- 2π,rs[I,end],".C0",markersize=8)

plot(xs, sum(w0'.*phi.(xs.-θ0'),dims=2), label=L"y_0", color="C1",":")

axis([0,2π,-15,15])
hlines(0,0,2π,"k")
xlabel(L"\mathcal{X}")
xticks([])
# yticks([0])
legend()
savefig("fig/mdpi_cpgd_fourier_1D.pdf",bbox_inches="tight")

