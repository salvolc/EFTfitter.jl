export EFTfitterModelWeights
export to_correlation_matrix

export get_observables, get_parameters, get_measurements, get_correlations
export get_nuisance_correlations, get_total_covariance, get_covariances, get_measurement_distributions

export EFTfitterDensityWeights


struct EFTfitterModelWeights 
    parameters::Any#BAT.NamedTupleDist
    measurements::NamedTuple{<:Any, <:Tuple{Vararg{Measurement}}}
    measurementdistributions::NamedTuple{<:Any, <:Tuple{Vararg{MeasurementDistribution}}}
    correlations::NamedTuple{<:Any, <:Tuple{Vararg{Correlation}}}
    weights::Vector{Int64}
    nuisances::Union{NamedTuple{<:Any, <:Tuple{Vararg{NuisanceCorrelation}}}, Nothing}
end

#include("/home/salv/.julia/dev/EFTfitter/src/EFTfitter.jl")
#include("/home/salv/.julia/dev/EFTfitter/src/datatypes.jl")
function EFTfitterModelWeights(
    parameters::Any,#BAT.NamedTupleDist,
    measurements::NamedTuple{<:Any, <:Tuple{Vararg{AbstractMeasurement}}},
    correlations::NamedTuple{<:Any, <:Tuple{Vararg{AbstractCorrelation}}},
    weights::Vector{Int64},
    nuisances::Union{NamedTuple{<:Any, <:Tuple{Vararg{NuisanceCorrelation}}}, Nothing} = nothing
)
    measurement_vec, measurement_keys = unpack(measurements)
    correlation_vec, uncertainty_keys = unpack(correlations)

    # convert elements of MeasurementDistribution to Measurement for each bin
    binned_measurements, binned_measurement_keys = convert_to_bins(measurement_vec, measurement_keys)
    # use only active measurements/bins
    active_measurements, active_measurement_keys, corrs = only_active_measurements(binned_measurements, binned_measurement_keys, correlation_vec)
    # use only active uncertainties and correlations
    active_measurements, active_correlations, uncertainty_keys = only_active_uncertainties(active_measurements, corrs, uncertainty_keys)

    correlation_nt = namedtuple(uncertainty_keys, active_correlations)
    measurement_nt = namedtuple(active_measurement_keys, active_measurements)
    meas_dists_nt  = create_distributions(measurements, uncertainty_keys)
    nuisances_nt   = only_active_nuisances(nuisances, active_measurement_keys, uncertainty_keys)
    
    #params = add_nuisance_parameters(parameters, nuisances_nt)
    params = parameters

    return EFTfitterModelWeights(params, measurement_nt, meas_dists_nt, correlation_nt, weights, nuisances_nt)
end

get_parameters(m::EFTfitterModelWeights) = m.parameters
get_measurements(m::EFTfitterModelWeights) = m.measurements
get_measurement_distributions(m::EFTfitterModelWeights) = m.measurementdistributions
get_correlations(m::EFTfitterModelWeights) = m.correlations
get_nuisance_correlations(m::EFTfitterModelWeights) = m.nuisances
function  get_observables(model::EFTfitterModelWeights)
    meas = get_measurements(model)
    obs = unique(Observable.([m.observable.func for m in values(meas)]))
    obs_names = [string(o.func) for o in obs]
    observables_nt = namedtuple(obs_names, obs)
end
function get_total_covariance(m::EFTfitterModelWeights)
    covs = get_covariances(m)
    total_cov = Symmetric(sum(covs))

    return total_cov
end
function get_covariances(m::EFTfitterModelWeights)
    unc_values = [[meas.uncertainties[u] for meas in m.measurements] for u in keys(m.correlations)]
    corrs = [c.matrix for c in m.correlations]

    covs = [Symmetric(σ*ρ*σ) for (σ, ρ) in zip(diagm.(unc_values), corrs)]
end
function has_nuisance_correlations(m::EFTfitterModelWeights)
    m.nuisances == nothing ? (return false) : (return true)
end


struct EFTfitterDensityWeights <: AbstractDensity
    measured_values::Vector{Float64}
    observable_functions::Vector{Function}
    observable_mins::Vector{Float64}
    observable_maxs::Vector{Float64}
    weights::Vector{Int64}
    sumw::Float64
    invcov::SparseMatrixCSC{Float64, Int64}
    check_bounds::Bool
end

function EFTfitterDensityWeights(m::EFTfitterModelWeights)
    n = length(m.measurements)
    measured_values = [meas.value for meas in m.measurements]
    observable_functions = [meas.observable.func for meas in m.measurements]
    observable_mins = [meas.observable.min for meas in m.measurements]
    observable_maxs = [meas.observable.max for meas in m.measurements]

    bu = any(x->x!=Inf, observable_maxs)
    bl = any(x->x!=-Inf, observable_mins)
    check_bounds = any([bu, bl])


    invcov = sparse(inv(get_total_covariance(m)))

    return EFTfitterDensityWeights(
            measured_values,
            observable_functions,
            observable_mins,
            observable_maxs,
            m.weights,
            sum(m.weights),
            invcov,
            check_bounds
            )
end




function BAT.PosteriorDensity(m::EFTfitterModelWeights)
    if has_nuisance_correlations(m)
        likelihood = EFTfitterDensityNuisance(m)
        return posterior = PosteriorDensity(likelihood, m.parameters)
    else
        likelihood = EFTfitterDensityWeights(m)
        return posterior = PosteriorDensity(likelihood, m.parameters)
    end
end

function BAT.eval_logval_unchecked(
    m::EFTfitterDensityWeights,
    params
)
    r = evaluate_funcs(m.observable_functions, params)

    if m.check_bounds
        ib = check_obs_bounds(r, m.observable_mins, m.observable_maxs)
        if ib == false
            return -Inf
        end
    end

    r = r-m.measured_values
    r1 = (m.invcov*r) .* m.weights
    result = -dot(r, r1) / m.sumw

    return  0.5*result
end