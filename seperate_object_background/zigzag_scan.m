function [vector] = zigzag_scan(M)
%zigzag scan a matrix, and out put a vector 
h = 1;
v = 1;
vmin = 1;
hmin = 1;
[vmax, hmax] = size(M);
i = 1;
vector = zeros(1,vmax * hmax);

while ((v <= vmax) && (h <= hmax))
    
    if (mod(h + v, 2) == 0)                 % going up
        if (v == vmin)       
            vector(i) = M(v, h);        % if we got to the first line
            if (h == hmax)
                v = v + 1;
            else
                h = h + 1;
            end
            i = i + 1;
        elseif ((h == hmax) && (v < vmax))   % if we got to the last column
            vector(i) = M(v, h);
            v = v + 1;
            i = i + 1;
        elseif ((v > vmin) && (h < hmax))    % all other cases
            vector(i) = M(v, h);
            v = v - 1;
            h = h + 1;
            i = i + 1;
        end
        
    else                                    % going down
       if ((v == vmax) && (h <= hmax))       % if we got to the last line
            vector(i) = M(v, h);
            h = h + 1;
            i = i + 1;
        
       elseif (h == hmin)                   % if we got to the first column
            vector(i) = M(v, h);
            if (v == vmax)
                h = h + 1;
            else
                v = v + 1;
            end
            i = i + 1;
       elseif ((v < vmax) && (h > hmin))     % all other cases
            vector(i) = M(v, h);
            v = v + 1;
            h = h - 1;
            i = i + 1;
       end
    end
    if ((v == vmax) && (h == hmax))          % bottom right element
        vector(i) = M(v, h);
        break
    end
end
end
