import PyPlot: gca, plt

function plot_seq(observations; states=[], changepoints=[], alpha=0.3, lw=1.0, ax=gca())
    ax[:plot](observations, lw=lw, drawstyle="steps-post")
    ax[:set_xlim](0, length(observations))

    if length(states) > 0
        cmap = Dict{Int,Tuple{Float64,Float64,Float64,Float64}}()
        for state in unique(states)
            cmap[state] = plt[:cm][:tab10](state)
        end

        last_idx, last_state = 1, states[1]
        for (i, state) in enumerate(states)
            if (state != last_state) || (i == length(states))
                ax[:axvspan](last_idx, i, alpha=alpha, color=cmap[last_state])
                last_idx, last_state = i, state
            end
        end
     end

    for changepoint in changepoints
        ax[:axvline](changepoint-1, color="red", alpha=0.5)
    end
end