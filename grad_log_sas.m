% Gradient computation of the - log SaS densities

function grad = grad_log_sas(v,alp)

addpath('./qastable');

grad = arrayfun(@(x) alp* stable_pdf_xder(x*alp,alp,0),v);
grad = grad ./ arrayfun(@(x) alp* stable_pdf(x*alp,alp,0),v);
grad = -grad;