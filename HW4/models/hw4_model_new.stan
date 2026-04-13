data {
    // Data for fitting
    int<lower=1> N;
    array[N] int<lower=0, upper=1> y;
    vector[N] x1;
    vector[N] x2;
    vector[N] x3;
    // Excluded data/testing data (not used for fitting)
    int<lower=0> N_ex;
    array[N_ex] int<lower=0, upper=1> y_ex;
    vector[N_ex] x1_ex;
    vector[N_ex] x2_ex;
    vector[N_ex] x3_ex;
}

parameters {
    real alpha;
    real beta1;
    real beta2;
    real beta3;
}

model {
    // Priors
    alpha ~ normal(0, 10);
    beta1 ~ normal(0, 5);
    beta2 ~ normal(0, 5);
    beta3 ~ normal(0, 5);

    // Likelihood
    y ~ bernoulli_logit(alpha + beta1 * x1 + beta2 * x2 + beta3 * x3);
}

generated quantities {
    // Fitting data
    vector[N] preds;
    vector[N] y_hat;
    for (n in 1:N) {
        preds[n] = bernoulli_logit_lpmf(y[n] | alpha + beta1 * x1[n] + beta2 * x2[n] + beta3 * x3[n]);
        y_hat[n] = bernoulli_logit_rng(alpha + beta1 * x1[n] + beta2 * x2[n] + beta3 * x3[n]);
    }

    // Excluded data
    vector[N_ex] preds_ex;
    for (n in 1:N_ex) {
        preds_ex[n] = bernoulli_logit_lpmf(y_ex[n] | alpha + beta1 * x1_ex[n] + beta2 * x2_ex[n] + beta3 * x3_ex[n]);
    }
}