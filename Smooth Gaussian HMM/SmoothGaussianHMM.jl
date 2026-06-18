# ============================================================
# Smooth Periodic Gaussian Hidden Markov Model
# ============================================================
#
# Goal:
#
# Implement a Smooth Periodic Hidden Markov Model with Gaussian
# emissions.
#
# The observed series is:
#
#     y_1, y_2, ..., y_N ∈ R
#
# Each observation has a periodic index:
#
#     τ_n ∈ {1, 2, ..., T}
#
# Example:
#
#     T = 12
#
# for monthly data.
#
# Hidden states:
#
#     Z_n ∈ {1, 2, ..., K}
#
# Gaussian emission:
#
#     Y_n | Z_n = k, τ_n = t
#         ~ Normal( μ_k(t), σ_k(t)^2 )
#
# Smooth periodic mean:
#
#     μ_k(t)
#         = β_{k,0}
#           + Σ_{r=1}^{d}
#             [
#               β^c_{k,r} cos(2π r t / T)
#               +
#               β^s_{k,r} sin(2π r t / T)
#             ]
#
# Smooth periodic log-standard deviation:
#
#     log σ_k(t)
#         = γ_{k,0}
#           + Σ_{r=1}^{d}
#             [
#               γ^c_{k,r} cos(2π r t / T)
#               +
#               γ^s_{k,r} sin(2π r t / T)
#             ]
#
# Therefore:
#
#     σ_k(t) = exp(log σ_k(t)) > 0
#
# Smooth transition probabilities:
#
#     P(Z_n = j | Z_{n-1} = i, τ_n = t)
#         = p_ij(t)
#
# Softmax parametrization:
#
#     p_ij(t)
#         =
#         exp(η_ij(t))
#         /
#         Σ_{l=1}^{K} exp(η_il(t))
#
# Smooth periodic transition logits:
#
#     η_ij(t)
#         = α_{ij,0}
#           + Σ_{r=1}^{d}
#             [
#               α^c_{ij,r} cos(2π r t / T)
#               +
#               α^s_{ij,r} sin(2π r t / T)
#             ]
#
# Identifiability constraint:
#
#     η_iK(t) = 0
#
# So we only store parameters for j = 1, ..., K-1.
#
# Parameter dimensions:
#
#     pi0              K
#     theta_A          K × (K - 1) × (2d + 1)
#     theta_mu         K × (2d + 1)
#     theta_logsigma   K × (2d + 1)
#
# ============================================================


using LinearAlgebra
using Statistics
using Distributions
using Random


# ============================================================
# Model structure
# ============================================================
#
# Formula represented by this struct:
#
#     θ = (π, α, β, γ)
#
# where:
#
#     π       = initial probabilities
#     α       = transition parameters
#     β       = Gaussian mean parameters
#     γ       = Gaussian log-standard-deviation parameters
#
# Dimensions:
#
#     pi0              K
#     theta_A          K × (K - 1) × (2d + 1)
#     theta_mu         K × (2d + 1)
#     theta_logsigma   K × (2d + 1)
#
# ============================================================

mutable struct SmoothGaussianHMM
    K::Int
    T::Int
    d::Int
    pi0::Vector{Float64}
    theta_A::Array{Float64,3}
    theta_mu::Matrix{Float64}
    theta_logsigma::Matrix{Float64}
end


# ============================================================
# Period index check
# ============================================================
#
# Mathematical condition:
#
#     τ_n ∈ {1, 2, ..., T}
#
# for every observation n.
#
# This function checks:
#
#     1 ≤ τ_n ≤ T
#
# ============================================================

function _sg_check_period_index(period_index::AbstractVector{<:Integer}, T::Int)
    all(t -> 1 <= t <= T, period_index) ||
        error("All period indices must be between 1 and T.")
end


# ============================================================
# Trigonometric basis
# ============================================================
#
# Formula:
#
#     x(t)
#         =
#         [
#           1,
#           cos(2π 1 t / T),
#           sin(2π 1 t / T),
#           ...,
#           cos(2π d t / T),
#           sin(2π d t / T)
#         ]
#
# Dimension:
#
#     length(x(t)) = 2d + 1
#
# This basis is used to construct:
#
#     μ_k(t)       = θ_mu[k]'       x(t)
#     log σ_k(t)  = θ_logsigma[k]' x(t)
#     η_ij(t)     = θ_A[i,j]'      x(t)
#
# ============================================================

function _sg_trig_basis(t::Int, T::Int, d::Int)
    basis = Float64[1.0]

    for r in 1:d
        push!(basis, cos(2π * r * t / T))
        push!(basis, sin(2π * r * t / T))
    end

    return basis
end


# ============================================================
# Design matrix
# ============================================================
#
# For a sequence of periodic indices:
#
#     τ_1, τ_2, ..., τ_N
#
# the design matrix is:
#
#     X =
#       [
#         x(τ_1)'
#         x(τ_2)'
#         ...
#         x(τ_N)'
#       ]
#
# Dimension:
#
#     X ∈ R^{N × (2d+1)}
#
# ============================================================

function _sg_design_matrix(period_index::AbstractVector{<:Integer}, T::Int, d::Int)
    _sg_check_period_index(period_index, T)

    p = 2d + 1
    X = zeros(length(period_index), p)

    for n in eachindex(period_index)
        X[n, :] .= _sg_trig_basis(period_index[n], T, d)
    end

    return X
end


# ============================================================
# Log-sum-exp
# ============================================================
#
# Formula:
#
#     log(Σ_i exp(a_i))
#
# Numerically stable version:
#
#     m = max_i a_i
#
#     log(Σ_i exp(a_i))
#         =
#         m + log(Σ_i exp(a_i - m))
#
# This avoids numerical overflow/underflow.
#
# ============================================================

function _sg_logsumexp(v::AbstractVector)
    m = maximum(v)
    return m + log(sum(exp.(v .- m)))
end


# ============================================================
# Weighted least squares
# ============================================================
#
# Used for approximate initialization / updates.
#
# Objective:
#
#     min_θ Σ_n w_n (y_n - x_n'θ)^2 + λ ||θ||²
#
# Matrix solution:
#
#     θ
#       =
#       (X' W X + λI)^{-1} X' W y
#
# where:
#
#     W = diag(w_1, ..., w_N)
#
# ============================================================

function _sg_weighted_lsq(
    X::Matrix{Float64},
    y::AbstractVector{<:Real},
    w::AbstractVector{<:Real};
    ridge::Float64 = 1e-6
)
    p = size(X, 2)

    if sum(w) <= 1e-12
        return zeros(p)
    end

    sw = sqrt.(max.(w, 0.0))

    Xw = X .* reshape(sw, :, 1)
    yw = collect(y) .* sw

    R = ridge * Matrix{Float64}(I, p, p)

    return (Xw' * Xw + R) \ (Xw' * yw)
end


# ============================================================
# Smooth periodic transition matrices
# ============================================================
#
# Transition formula:
#
#     P(Z_n = j | Z_{n-1} = i, τ_n = t)
#         = p_ij(t)
#
# Softmax:
#
#     p_ij(t)
#         =
#         exp(η_ij(t))
#         /
#         Σ_{l=1}^{K} exp(η_il(t))
#
# Smooth transition logit:
#
#     η_ij(t) = θ_A[i,j]' x(t)
#
# Reference category:
#
#     η_iK(t) = 0
#
# Therefore, theta_A stores only:
#
#     j = 1, ..., K-1
#
# Output:
#
#     A[i,j,t] = p_ij(t)
#
# Dimension:
#
#     A ∈ R^{K × K × T}
#
# ============================================================

function smooth_gaussian_transition_matrices(model::SmoothGaussianHMM)
    K = model.K
    T = model.T
    d = model.d

    A = zeros(K, K, T)

    for t in 1:T
        bt = _sg_trig_basis(t, T, d)

        for i in 1:K
            logits = zeros(K)

            for j in 1:(K - 1)
                logits[j] = dot(view(model.theta_A, i, j, :), bt)
            end

            logits[K] = 0.0

            m = maximum(logits)
            probs = exp.(logits .- m)

            A[i, :, t] .= probs ./ sum(probs)
        end
    end

    return A
end


# ============================================================
# Smooth Gaussian emission parameters
# ============================================================
#
# Gaussian emission:
#
#     Y_n | Z_n = k, τ_n = t
#         ~ Normal( μ_k(t), σ_k(t)^2 )
#
# Smooth mean:
#
#     μ_k(t) = θ_mu[k]' x(t)
#
# Smooth log-standard deviation:
#
#     log σ_k(t) = θ_logsigma[k]' x(t)
#
# Therefore:
#
#     σ_k(t) = exp(θ_logsigma[k]' x(t))
#
# This guarantees:
#
#     σ_k(t) > 0
#
# Outputs:
#
#     mu[k,t]     = μ_k(t)
#     sigma[k,t]  = σ_k(t)
#
# ============================================================

function smooth_gaussian_emission_parameters(model::SmoothGaussianHMM)
    K = model.K
    T = model.T
    d = model.d

    mu = zeros(K, T)
    sigma = zeros(K, T)

    for k in 1:K
        for t in 1:T
            bt = _sg_trig_basis(t, T, d)

            mu[k, t] = dot(view(model.theta_mu, k, :), bt)
            sigma[k, t] = exp(dot(view(model.theta_logsigma, k, :), bt))
        end
    end

    return mu, sigma
end


# ============================================================
# Gaussian emission log-likelihoods
# ============================================================
#
# Gaussian density:
#
#     b_k(y_n, τ_n)
#         =
#         f_Normal(y_n ; μ_k(τ_n), σ_k(τ_n)^2)
#
# Log-density:
#
#     log b_k(y_n, τ_n)
#         =
#         - 1/2 log(2π)
#         - log σ_k(τ_n)
#         - (y_n - μ_k(τ_n))² / (2 σ_k(τ_n)²)
#
# Output matrix:
#
#     LL[n,k] = log b_k(y_n, τ_n)
#
# Dimension:
#
#     LL ∈ R^{N × K}
#
# ============================================================

function smooth_gaussian_emission_loglikelihoods(
    model::SmoothGaussianHMM,
    y::AbstractVector{<:Real},
    period_index::AbstractVector{<:Integer}
)
    length(y) == length(period_index) ||
        error("y and period_index must have the same length.")

    _sg_check_period_index(period_index, model.T)

    K = model.K
    N = length(y)

    mu, sigma = smooth_gaussian_emission_parameters(model)

    LL = zeros(N, K)

    for n in 1:N
        t = period_index[n]

        for k in 1:K
            LL[n, k] = logpdf(Normal(mu[k, t], sigma[k, t]), y[n])
        end
    end

    return LL
end


# ============================================================
# Model initialization
# ============================================================
#
# Initial state allocation:
#
#     y_n is split into K empirical quantile groups.
#
# This gives an initial rough regime classification:
#
#     initial_state_n ∈ {1, ..., K}
#
# Mean initialization:
#
#     μ_k(t) = θ_mu[k]' x(t)
#
# θ_mu[k] is estimated by least squares using observations
# initially assigned to state k.
#
# Variance initialization:
#
#     σ_k ≈ std(y_n - μ_k(τ_n))
#
# Transition initialization:
#
# Persistent transition matrix:
#
#     P0[k,k] = p_stay
#
#     P0[i,j] = (1 - p_stay) / (K - 1),   i ≠ j
#
# Convert transition probabilities to logits:
#
#     θ_A[i,j,1] = log(P0[i,j] / P0[i,K])
#
# because the last state is the reference category.
#
# ============================================================

function initialize_smooth_gaussian_hmm(
    y::AbstractVector{<:Real},
    period_index::AbstractVector{<:Integer};
    K::Int = 3,
    T::Int = 12,
    d::Int = 1,
    p_stay::Float64 = 0.90,
    ridge::Float64 = 1e-6
)
    K >= 2 || error("K must be at least 2.")
    0.0 < p_stay < 1.0 || error("p_stay must be between 0 and 1.")
    length(y) == length(period_index) ||
        error("y and period_index must have the same length.")

    _sg_check_period_index(period_index, T)

    y_vec = collect(Float64, y)

    p = 2d + 1
    N = length(y_vec)

    probs = collect(range(0, 1, length = K + 1))[2:K]
    thresholds = quantile(y_vec, probs)

    initial_state = Vector{Int}(undef, N)

    for n in 1:N
        initial_state[n] = searchsortedfirst(thresholds, y_vec[n]) + 1
    end

    X = _sg_design_matrix(period_index, T, d)

    theta_mu = zeros(K, p)
    theta_logsigma = zeros(K, p)

    for k in 1:K
        idx = findall(==(k), initial_state)

        if length(idx) >= p
            theta_mu[k, :] .= _sg_weighted_lsq(
                X[idx, :],
                y_vec[idx],
                ones(length(idx));
                ridge = ridge
            )
        else
            theta_mu[k, 1] = mean(y_vec)
        end
    end

    fitted = zeros(N)

    for n in 1:N
        k = initial_state[n]
        fitted[n] = dot(view(theta_mu, k, :), X[n, :])
    end

    for k in 1:K
        idx = findall(==(k), initial_state)

        if length(idx) >= 2
            s = std(y_vec[idx] .- fitted[idx])
        else
            s = std(y_vec)
        end

        theta_logsigma[k, 1] = log(max(s, 0.10))
    end

    P0 = fill((1.0 - p_stay) / (K - 1), K, K)

    for k in 1:K
        P0[k, k] = p_stay
    end

    theta_A = zeros(K, K - 1, p)

    for i in 1:K
        for j in 1:(K - 1)
            theta_A[i, j, 1] = log(P0[i, j] / P0[i, K])
        end
    end

    pi0 = fill(1.0 / K, K)

    return SmoothGaussianHMM(
        K,
        T,
        d,
        pi0,
        theta_A,
        theta_mu,
        theta_logsigma
    )
end


# ============================================================
# Forward algorithm in log scale
# ============================================================
#
# Forward variable:
#
#     α_n(k)
#         =
#         P(y_1, ..., y_n, Z_n = k)
#
# Initialization:
#
#     α_1(k)
#         =
#         π_k b_k(y_1, τ_1)
#
# Recursion:
#
#     α_n(j)
#         =
#         b_j(y_n, τ_n)
#         Σ_{i=1}^{K}
#         α_{n-1}(i) p_ij(τ_n)
#
# Log-scale version:
#
#     log α_n(j)
#         =
#         log b_j(y_n, τ_n)
#         +
#         log Σ_i exp[
#             log α_{n-1}(i)
#             +
#             log p_ij(τ_n)
#         ]
#
# Log-likelihood:
#
#     log L(θ)
#         =
#         log Σ_k exp(log α_N(k))
#
# ============================================================

function smooth_gaussian_forward_log(
    model::SmoothGaussianHMM,
    y::AbstractVector{<:Real},
    period_index::AbstractVector{<:Integer}
)
    A = smooth_gaussian_transition_matrices(model)
    LL = smooth_gaussian_emission_loglikelihoods(model, y, period_index)

    N, K = size(LL)

    log_alpha = zeros(N, K)
    log_pi0 = log.(model.pi0)

    for k in 1:K
        log_alpha[1, k] = log_pi0[k] + LL[1, k]
    end

    for n in 2:N
        t = period_index[n]

        for j in 1:K
            terms = [
                log_alpha[n - 1, i] + log(A[i, j, t])
                for i in 1:K
            ]

            log_alpha[n, j] = LL[n, j] + _sg_logsumexp(terms)
        end
    end

    loglik = _sg_logsumexp(log_alpha[N, :])

    return log_alpha, loglik, LL, A
end


# ============================================================
# Log-likelihood
# ============================================================
#
# Formula:
#
#     log L(θ)
#         =
#         log Σ_k exp(log α_N(k))
#
# This function only returns the scalar log-likelihood.
#
# ============================================================

function smooth_gaussian_loglikelihood(
    model::SmoothGaussianHMM,
    y::AbstractVector{<:Real},
    period_index::AbstractVector{<:Integer}
)
    _, loglik, _, _ = smooth_gaussian_forward_log(model, y, period_index)
    return loglik
end


# ============================================================
# Backward algorithm in log scale
# ============================================================
#
# Backward variable:
#
#     β_n(i)
#         =
#         P(y_{n+1}, ..., y_N | Z_n = i)
#
# Terminal condition:
#
#     β_N(i) = 1
#
# Log-scale:
#
#     log β_N(i) = 0
#
# Recursion:
#
#     β_n(i)
#         =
#         Σ_{j=1}^{K}
#         p_ij(τ_{n+1})
#         b_j(y_{n+1}, τ_{n+1})
#         β_{n+1}(j)
#
# Log-scale recursion:
#
#     log β_n(i)
#         =
#         log Σ_j exp[
#             log p_ij(τ_{n+1})
#             +
#             log b_j(y_{n+1}, τ_{n+1})
#             +
#             log β_{n+1}(j)
#         ]
#
# ============================================================

function smooth_gaussian_backward_log(
    model::SmoothGaussianHMM,
    y::AbstractVector{<:Real},
    period_index::AbstractVector{<:Integer},
    LL::Matrix{Float64},
    A::Array{Float64,3}
)
    N, K = size(LL)

    log_beta = zeros(N, K)

    log_beta[N, :] .= 0.0

    for n in (N - 1):-1:1
        t_next = period_index[n + 1]

        for i in 1:K
            terms = [
                log(A[i, j, t_next]) + LL[n + 1, j] + log_beta[n + 1, j]
                for j in 1:K
            ]

            log_beta[n, i] = _sg_logsumexp(terms)
        end
    end

    return log_beta
end


# ============================================================
# Posterior probabilities
# ============================================================
#
# State posterior:
#
#     γ_n(k)
#         =
#         P(Z_n = k | y_1, ..., y_N)
#
# Forward-backward formula:
#
#     γ_n(k)
#         =
#         α_n(k) β_n(k) / L(θ)
#
# Log-scale:
#
#     γ_n(k)
#         =
#         exp(
#             log α_n(k)
#             +
#             log β_n(k)
#             -
#             log L(θ)
#         )
#
# Transition posterior:
#
#     ξ_n(i,j)
#         =
#         P(Z_{n-1}=i, Z_n=j | y_1, ..., y_N)
#
# Formula:
#
#     ξ_n(i,j)
#         ∝
#         α_{n-1}(i)
#         p_ij(τ_n)
#         b_j(y_n, τ_n)
#         β_n(j)
#
# Log-scale:
#
#     ξ_n(i,j)
#         =
#         exp(
#             log α_{n-1}(i)
#             +
#             log p_ij(τ_n)
#             +
#             log b_j(y_n, τ_n)
#             +
#             log β_n(j)
#             -
#             log L(θ)
#         )
#
# ============================================================

function smooth_gaussian_posteriors(
    model::SmoothGaussianHMM,
    y::AbstractVector{<:Real},
    period_index::AbstractVector{<:Integer}
)
    log_alpha, loglik, LL, A = smooth_gaussian_forward_log(model, y, period_index)
    log_beta = smooth_gaussian_backward_log(model, y, period_index, LL, A)

    N, K = size(LL)

    gamma = zeros(N, K)

    for n in 1:N
        for k in 1:K
            gamma[n, k] = exp(log_alpha[n, k] + log_beta[n, k] - loglik)
        end

        gamma[n, :] ./= sum(gamma[n, :])
    end

    xi = zeros(N - 1, K, K)

    for n in 2:N
        t = period_index[n]

        for i in 1:K
            for j in 1:K
                xi[n - 1, i, j] =
                    exp(
                        log_alpha[n - 1, i] +
                        log(A[i, j, t]) +
                        LL[n, j] +
                        log_beta[n, j] -
                        loglik
                    )
            end
        end

        total = sum(xi[n - 1, :, :])

        if total > 0
            xi[n - 1, :, :] ./= total
        end
    end

    return gamma, xi, loglik
end


# ============================================================
# Viterbi algorithm
# ============================================================
#
# Goal:
#
#     ẑ_{1:N}
#         =
#         argmax_{z_{1:N}}
#         P(z_{1:N} | y_{1:N})
#
# Dynamic programming variable:
#
#     δ_n(j)
#         =
#         max_{z_1,...,z_{n-1}}
#         log P(z_1,...,z_{n-1}, Z_n=j, y_1,...,y_n)
#
# Initialization:
#
#     δ_1(k)
#         =
#         log π_k + log b_k(y_1, τ_1)
#
# Recursion:
#
#     δ_n(j)
#         =
#         log b_j(y_n, τ_n)
#         +
#         max_i [
#             δ_{n-1}(i)
#             +
#             log p_ij(τ_n)
#         ]
#
# Backtracking:
#
#     ψ_n(j)
#         =
#         argmax_i [
#             δ_{n-1}(i)
#             +
#             log p_ij(τ_n)
#         ]
#
# ============================================================

function smooth_gaussian_viterbi(
    model::SmoothGaussianHMM,
    y::AbstractVector{<:Real},
    period_index::AbstractVector{<:Integer}
)
    A = smooth_gaussian_transition_matrices(model)
    LL = smooth_gaussian_emission_loglikelihoods(model, y, period_index)

    N, K = size(LL)

    delta = zeros(N, K)
    psi = zeros(Int, N, K)

    log_pi0 = log.(model.pi0)

    for k in 1:K
        delta[1, k] = log_pi0[k] + LL[1, k]
    end

    for n in 2:N
        t = period_index[n]

        for j in 1:K
            values = [
                delta[n - 1, i] + log(A[i, j, t])
                for i in 1:K
            ]

            psi[n, j] = argmax(values)
            delta[n, j] = maximum(values) + LL[n, j]
        end
    end

    z_hat = zeros(Int, N)
    z_hat[N] = argmax(delta[N, :])

    for n in (N - 1):-1:1
        z_hat[n] = psi[n + 1, z_hat[n + 1]]
    end

    return z_hat
end


# ============================================================
# Simulation
# ============================================================
#
# Simulation procedure:
#
# Initial state:
#
#     Z_1 ~ Categorical(pi0)
#
# First observation:
#
#     Y_1 | Z_1=k, τ_1=t
#         ~ Normal(μ_k(t), σ_k(t)^2)
#
# State recursion:
#
#     Z_n | Z_{n-1}=i, τ_n=t
#         ~ Categorical(A[i,:,t])
#
# Observation recursion:
#
#     Y_n | Z_n=k, τ_n=t
#         ~ Normal(μ_k(t), σ_k(t)^2)
#
# ============================================================

function smooth_gaussian_rand(
    rng::AbstractRNG,
    model::SmoothGaussianHMM,
    N::Int;
    period_index::Union{Nothing,AbstractVector{<:Integer}} = nothing
)
    if period_index === nothing
        period_index = [mod(n - 1, model.T) + 1 for n in 1:N]
    end

    length(period_index) == N ||
        error("period_index must have length N.")

    _sg_check_period_index(period_index, model.T)

    A = smooth_gaussian_transition_matrices(model)
    mu, sigma = smooth_gaussian_emission_parameters(model)

    z = zeros(Int, N)
    y = zeros(Float64, N)

    z[1] = rand(rng, Categorical(model.pi0))

    t = period_index[1]
    y[1] = rand(rng, Normal(mu[z[1], t], sigma[z[1], t]))

    for n in 2:N
        t = period_index[n]

        z[n] = rand(rng, Categorical(A[z[n - 1], :, t]))
        y[n] = rand(rng, Normal(mu[z[n], t], sigma[z[n], t]))
    end

    return y, z
end


function smooth_gaussian_rand(
    model::SmoothGaussianHMM,
    N::Int;
    period_index::Union{Nothing,AbstractVector{<:Integer}} = nothing
)
    return smooth_gaussian_rand(
        Random.default_rng(),
        model,
        N;
        period_index = period_index
    )
end