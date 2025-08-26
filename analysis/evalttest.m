%function [outputArg1,outputArg2] = untitled(inputArg1,inputArg2)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%outputArg1 = inputArg1;
%outputArg2 = inputArg2;
%end


function evaluate_ttest(H,P,CI,X,Y)
%evalttest Used to interpret the results of a two-sample t-test.
%   Detailed explanation goes here
    if H==1
    fprintf(['We are 95%% sure that the means of %s and %s are\n' ...
        'significantly different and that the true difference\n' ...
        'between the means is between %.2f and %.2f.\n'], X, Y, CI(1), CI(2));
    else
        fprintf(['We are 95%% sure that the means of %s and %s are\n' ...
            'not significantly different and that the true difference\n' ...
            'between the means is between %.2f and %.2f.\n'], X, Y, CI(1), CI(2));
    end
    
    if CI(1) < 0 && CI(2) < 0
        fprintf(['Based on the confidence interval, the mean of %s\n' ...
            'is significantly lower than the mean of %s.'], X, Y);
    else if CI(1) > 0 && CI(2) > 0
        fprintf(['Based on the confidence interval, the mean of %s\n' ...
            'is significantly higher than the mean of %s.'], X, Y);
    end
end
