function [freq_mat] = extract_features(bspec, axes)

    % FREQUENCY BANDS HARCODED ATM

    % δ (0.5 − 4Hz) 
    % θ (5 − 7Hz)
    % α (8 − 15Hz)
    % β (16 − 31Hz
    % γ (32 − 100Hz)
    
    % at 2k hz sampling
    % 1000hz -> 1
    % 100hz -> 0.1
    % 32hz -> 0.032
    
    % at 500 hz sampling
    % 100hz -> 0.25
    % 32hz -> 0.08
    % 16hz -> 0.04
    % 8hz -> 0.02
    % 4hz -> 0.01
    % 0.5hz -> 0.00125
    
%     disp(min_index(axes, 0.25))
    
    freq_mat = zeros(5, 5);
    
    t1 = min_index(axes, 0.00125);
    t2 = min_index(axes, 0.01);
    t3 = min_index(axes, 0.02);
    t4 = min_index(axes, 0.04);
    t5 = min_index(axes, 0.08);
    t6 = min_index(axes, 0.25);
    
    % δ - x
    freq_mat(1,1) = max(max(real(bspec(t1:t2-1, t1:t2-1)))); 
    freq_mat(1,2) = max(max(real(bspec(t1:t2-1, t2:t3-1))));
    freq_mat(1,3) = max(max(real(bspec(t1:t2-1, t3:t4-1))));
    freq_mat(1,4) = max(max(real(bspec(t1:t2-1, t4:t5-1))));
    freq_mat(1,5) = max(max(real(bspec(t1:t2-1, t5:t6-1))));

    % θ - x
    freq_mat(2,1) = max(max(real(bspec(t2:t3-1, t1:t2-1))));
    freq_mat(2,2) = max(max(real(bspec(t2:t3-1, t2:t3-1))));
    freq_mat(2,3) = max(max(real(bspec(t2:t3-1, t3:t4-1))));
    freq_mat(2,4) = max(max(real(bspec(t2:t3-1, t4:t5-1))));
    freq_mat(2,5) = max(max(real(bspec(t2:t3-1, t5:t6-1))));

    % α - x
    freq_mat(3,1) = max(max(real(bspec(t3:t4-1, t1:t2-1))));
    freq_mat(3,2) = max(max(real(bspec(t3:t4-1, t2:t3-1))));
    freq_mat(3,3) = max(max(real(bspec(t3:t4-1, t3:t4-1))));
    freq_mat(3,4) = max(max(real(bspec(t3:t4-1, t4:t5-1))));
    freq_mat(3,5) = max(max(real(bspec(t3:t4-1, t5:t6-1))));
    
    % β - x
    freq_mat(4,1) = max(max(real(bspec(t4:t5-1, t1:t2-1))));
    freq_mat(4,2) = max(max(real(bspec(t4:t5-1, t2:t3-1))));
    freq_mat(4,3) = max(max(real(bspec(t4:t5-1, t3:t4-1))));
    freq_mat(4,4) = max(max(real(bspec(t4:t5-1, t4:t5-1))));
    freq_mat(4,5) = max(max(real(bspec(t4:t5-1, t5:t6-1))));

    % γ - x
    freq_mat(5,1) = max(max(real(bspec(t5:t6-1, t1:t2-1))));
    freq_mat(5,2) = max(max(real(bspec(t5:t6-1, t2:t3-1))));
    freq_mat(5,3) = max(max(real(bspec(t5:t6-1, t3:t4-1))));
    freq_mat(5,4) = max(max(real(bspec(t5:t6-1, t4:t5-1))));
    freq_mat(5,5) = max(max(real(bspec(t5:t6-1, t5:t6-1))));

end

