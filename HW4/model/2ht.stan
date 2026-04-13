data {
    int N_old;
    int N_new;
    int hits;
    int misses;
    int FA;
    int CR;
}

parameters {
    real<lower = 0, upper = 1> d;
    real<lower = 0, upper = 1> g;
}

model {
    // Priors
    d ~ beta(1, 1);
    g ~ beta(1, 1);
    
    // Likelihood
    hits ~ binomial(N_old, d + g * (1 - d));
    misses ~ binomial(N_old, (1 - d) * (1 - g));
    FA ~ binomial(N_new, (1 - d) * g);
    CR ~ binomial(N_new, d + (1 - d) * (1 - g));
}

