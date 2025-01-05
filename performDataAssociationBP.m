% Florian Meyer, 2017

function [outputLegacy, outputNew] = performDataAssociationBP(inputLegacy, inputNew, checkConvergence, threshold, numIterations )
[numMeasurements,numObjects] = size(inputLegacy);
numMeasurements = numMeasurements-1;

outputLegacy = ones(numMeasurements,numObjects);
outputNew = ones(numMeasurements,1);

if(numObjects == 0 || numMeasurements == 0)
    return;
end

om = ones(1,numMeasurements);
on = ones(1,numObjects);
messages2 = ones(numMeasurements,numObjects);

% missed detection does not depend on the measurement
% therefore no need to update it
% in the normalization step, the missed detection is added separately

for iteration = 1:numIterations
    messages2Old = messages2;
    % the multiplication in the denominator of equation 30
    product1 = messages2 .* inputLegacy(2:end,:);
    % demoninator of equation 30
    sum1 = inputLegacy(1,:) + sum(product1,1);

    messages1 = inputLegacy(2:end,:) ./ (sum1(om,:) - product1);
    sum2 = inputNew + sum(messages1,2);
    messages2 = 1 ./ (sum2(:,on) - messages1);
  
    if(mod(iteration,checkConvergence) == 0)
        distance = max(max(abs(log(messages2./messages2Old))));
        if(distance < threshold)
            break
        end
    end
end


outputLegacy = messages2;

outputNew = [ones(numMeasurements,1),messages1];
outputNew = outputNew./repmat(sum(outputNew,2),[1,numObjects+1]);
outputNew = outputNew(:,1);

end