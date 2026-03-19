data {
    int<lower=1> N; // number of rows
    int<lower=1> J; // number of people
    array[N] int<lower=1, upper=J> id;   // person ID for each trial/row
    array[N] real<lower=0> y; // response time
    array[N] int<lower=1, upper=2> condition; // condition indicator
    array[N] int<lower=0, upper=1> choice; // choice indicator
}

transformed data {
  // init a vector to hold the minimum response times
  real eps = 1e-6;
  array[J] real rt_min;

  // initialize to +inf
  for (j in 1:J) {
    rt_min[j] = positive_infinity();
  }

  // compute per-person minimum RT
  for (n in 1:N) {
    if (y[n] < rt_min[id[n]]) {
        rt_min[id[n]] = y[n];
    }
  }
}

parameters {
    // Each person has their own parameters
    // Your code here: One of these parameters should index the
    // difficulty of the task. Which one? You will then have to
    // assume two parameters vectors, one for each condition. :)
    array[J, 2] real v;
    array[J] real<lower=0> a;
    array[J] real<lower=0, upper=1> beta;
    array[J] real<lower=0, upper=1> tau_raw;
}

transformed parameters {
  // A good way to bound non-decision time (tau)
  array[J] real<lower=0> tau;
  for (j in 1:J) {
    tau[j] = tau_raw[j] * (rt_min[j] - eps);
  }
}

model {
    // Priors
    // Your code here

    // Likelihood
    for (n in 1:N) {
        // Condition 1
        if (condition[n] == 1) {
            if (choice[n] == 1) {
                // Your code here 
                // Hint: use something like param[id[n]] 
                // to index the person-specific parameters ;)
            } else {
                // Your code here
            }
        }

        // Condition 2
        if (condition[n] == 2) {
            if (choice[n] == 1) {
                // Your code here
            } else {
                // Your code here
            }
        }
    }
}