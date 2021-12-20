data {
    int N;
    matrix[N, 2] X;
    int<lower=0, upper=1> y[N];
}
 
parameters {
     vector[2] w;
}

model {
    w ~ normal(0, 1);
    y ~ bernoulli_logit(X * w);
}    