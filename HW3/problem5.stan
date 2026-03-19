data{
    int<lower = 1> N; 
    vector[N] x; 
    vector[N] y;
}

parameters{
    real<lower= 0> sigma2;
    real alpha;
    real beta;
}

transformed parameters{
    real sigma;
    sigma = sqrt(sigma2);
} 

model{
    alpha ~ normal(0, 10);
    beta ~ normal(0, 10);
    sigma2 ~ inv_gamma(1, 1);
    for(n in 1:N){
        y[n] ~ normal(alpha + beta * x[n], sigma);
    }
}