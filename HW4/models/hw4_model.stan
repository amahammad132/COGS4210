data {
    int<lower=1> N;
    int<lower=1> N_tilde;
    array[N] int<lower=0, upper=1> y;
    vector[N] x1;
    vector[N] x2;
    vector[N] x3;
    vector[N_tilde] x_tilde;
}

parameters {
    real<lower=0> sigma;
    real alpha;
    real beta;
}

model {
    // Priors
    sigma ~ cauchy(0, 5);
    alpha ~ cauchy(0, 10);
    beta ~ normal(0, 10);

    // Likelihood
    y ~ normal(alpha + beta * x, sigma);
    // for (n in 1:N) {
    //     if (cloze[n] == 1) {
    //         signal[n] ~ normal(alpha + beta, sigma);
    //     } else {
    //         signal[n] ~ normal(alpha - beta, sigma);
    //     }
    //     // signal[n] ~ normal(alpha + cloze[n] * beta, sigma);

    // }
}

generated quantities {
    vector[N_tilde] y_tilde;
    array[N_tilde] int<lower=0, upper=1> pred_dec;
    for (n in 1:N_tilde) {
        y_tilde[n] = normal_rng(alpha + beta * x_tilde[n], sigma);
        if (y_tilde[n] > (alpha + beta)) {
            pred_dec[n] = 1;
        } else {
            pred_dec[n] = 0;
        }
    }
}