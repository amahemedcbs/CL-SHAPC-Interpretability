function diff_norms = calc_diff_norms(maps1, inclass, maps2)
%--------------------------------------------------------------------
% Used to calculate the L2 norms of dct differences.
%
% function [diff_norms] = calc_diff_norms(dcts1,dcts2)
% pdf1: first pdf to compare
% pdf2: second pdf to compare
% example: [result]=findfeatures('s1.pgm',35); 
%
%--------------------------------------------------------------------

    if ~exist('inclass','var')
        inclass=true;
    end
    
    norm_attrs1 = [];
    norm_attrs2 = [];
    diff_norms = [];
    
    for i=1:length(fieldnames(maps1))
        map_str = sprintf('map%d', i-1);
        grad1 = squeeze(maps1.(map_str).grad);
        if size(grad1,1)==28
            grad1 = reshape(grad1, [1 28 28]);
        end
        norm_attr1 = squeeze(normalize_attr(grad1, 0));
        norm_attrs1 = [norm_attrs1; norm_attr1];
        
        if inclass ~= true
            grad2 = squeeze(maps2.(map_str).grad);
            if size(grad2,1)==28
                grad2 = reshape(grad2, [1 28 28]);
            end
            norm_attr2 = squeeze(normalize_attr(grad2, 0));
            norm_attrs2 = [norm_attrs2; norm_attr2];
        end    
        
    end
    
    dim = size(grad1,2);
    dctlength = dim^2 - 1;
    %dctlength = (dim^2) / 2;
    %dctlength = 256;
    sz = size(norm_attrs1, 1)/dim;
    if inclass==true
        for i=1:sz
            for j=i+1:sz
                %test = norm_attrs1((i*dim)+1:(i*dim)+dim,:)
                zigi = findfeatures(norm_attrs1(((i-1)*dim)+1:((i-1)*dim)+dim,:), dctlength);
                zigj = findfeatures(norm_attrs1(((j-1)*dim)+1:((j-1)*dim)+dim,:), dctlength);
                diff = norm(zigi-zigj);
                diff_norms = [diff_norms diff];
            end
        end
    else
        for i=1:sz
            for j=1:sz
                zigi = findfeatures(norm_attrs1(((i-1)*dim)+1:((i-1)*dim)+dim,:), dctlength);
                zigj = findfeatures(norm_attrs2(((j-1)*dim)+1:((j-1)*dim)+dim,:), dctlength);
                diff = norm(zigi-zigj);
                diff_norms = [diff_norms diff];
            end
        end
    end

end
