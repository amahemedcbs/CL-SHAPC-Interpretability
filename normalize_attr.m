function norm_attr = normalize_attr(attr, reduction_axis)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

    outlier_perc = 2;
    attr_combined = attr;
    if ~exist('reductionaxis','var')
        attr_combined = sum(attr, reduction_axis+1);
    end

    attr_combined = abs(attr_combined);
    threshold = cumulative_sum_threshold(attr_combined, 100 - outlier_perc);
    norm_attr = normalize_scale(attr_combined, threshold);
end

function threshold = cumulative_sum_threshold(values, percentile)
    %sorted_vals = sort(values.flatten());
    sorted_vals = sort(values(:));
    cum_sums = cumsum(sorted_vals);

    %{
    threshold_id1 = where(cum_sums >= cum_sums(end) * 0.01 * percentile);
    threshold_id1 = cum_sums(cum_sums >= cum_sums(end) * 0.01 * percentile)
    threshold_id2 = threshold_id1(1)
    threshold_id = threshold_id2(1)
    threshold = sorted_vals(threshold_id+1);
    %}

    % Find the index where the cumulative sum reaches the percentile threshold
    % MATLAB's find returns all indices, so we take the first one.
    threshold_id_relative = find(cum_sums >= cum_sums(end) * 0.01 * percentile, 1, 'first');
    
    % The index in MATLAB is 1-based, so no need for [0][0] like Python
    threshold = sorted_vals(threshold_id_relative);
end

function norm_attr = normalize_scale(attr, scale_factor)
    attr_norm = attr / scale_factor;
    norm_attr = clip(attr_norm, -1, 1);
end

