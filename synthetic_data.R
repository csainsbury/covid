## COVID synthetic data
library(data.table)
n = 100000

continuousFeatureTable <- as.data.table(matrix(0, nrow = n, ncol = 0))

unique_ID <- c(1:n)
unique_ID <- unique_ID + (n * 10)

dmt1_prev <- 6
dmt2_prev <- 30
cvd_prev  <- 40
copd_prev <- 25
obes_prev <- 40

sglt2_use <- 20 # (if dmt2 == 1)
ace_use   <- 60 # (if bp == 1)

## biochem
## Na, K, Cl, Ur, Creat
na = rnorm(n, mean= 140, sd = 4.5)
cl = rnorm(n, mean= 102, sd = 4) # 98-106
k  = rnorm(n, mean= 4.2, sd = 1) # 3.5 - 5
urea = rnorm(n, mean= 5, sd = 2) # 2.9-7.1
creat = rnorm(n, mean= 180, sd = 40) # 90 -300

glu = rgamma(n, 3, 0.5)

ast = rnorm(n, mean= 70, sd = 30) # 90 -300
alt = rnorm(n, mean= 65, sd = 40) # 90 -300
alb = rnorm(n, mean= 38, sd = 4)

Ca = rnorm(n, mean= 2.4, sd = 0.2)
aCa = ifelse(alb > 40, Ca - ((40 - alb) * 0.02), Ca + ((40 - alb) * 0.02))
Po4 = rnorm(n, mean= 1.6, sd = 0.2)

crp = rgamma(n, 2, 0.10)

hb = rnorm(n, mean= 120, sd = 10)
mcv = rnorm(n, mean= 90, sd = 4)
neut = rgamma(n, 2, 0.4)
lymph = rgamma(n, 1.2, 0.6)
eosin = rgamma(n, 0.2, 0.8)
wcc = neut + lymph + eosin + rgamma(n, 0.1, 0.8)

## add small amount aldo excess
aldoflag <- ifelse(na > 146, sample(c(rep(0, round(100 / 50, 1)), 1), n, replace = TRUE), 0)
random_aldoFactor <- rnorm(n, 0.6, 0.2)
k <- ifelse(aldoflag == 1, k * random_aldoFactor, k)
k = ifelse(k < 1, 1, k)

## hypertension chance of bp +ve going up with cube of na
na3 <- na^3
na3 <- (na3 - min(na3)) / (max(na3) - min(na3))

## some demogs and categoricals
age <- rweibull(n, 4, 64)
sex <- sample(c(0, 1), n, replace = TRUE)

## BP
bp <- sample(c(0, 0, 0, 1), n, replace = TRUE)
bp <- ifelse(bp == 0 & na3 > 0.45, sample(c(rep(0, round(100 / 40, 1)), 1), n, replace = TRUE), bp)
dmt1 <- sample(c(rep(0, round(100 / dmt1_prev, 1)), 1), n, replace = TRUE)
dmt2 <- sample(c(rep(0, round(100 / dmt2_prev, 1)), 1), n, replace = TRUE)
cvd <- sample(c(rep(0, round(100 / cvd_prev, 1)), 1), n, replace = TRUE)
copd <- sample(c(rep(0, round(100 / copd_prev, 1)), 1), n, replace = TRUE)
obes <- sample(c(rep(0, round(100 / obes_prev, 1)), 1), n, replace = TRUE)

## drug
## BP

sglt2 <- ifelse(dmt2 == 1, sample(c(rep(0, round(100 / sglt2_use, 1)), 1), n, replace = TRUE), 0)
ace <- ifelse(bp == 1, sample(c(rep(0, round(100 / ace_use, 1)), 1), n, replace = TRUE), 0)
ace <- ifelse(dmt2 == 1, sample(c(rep(0, round(100 / ace_use, 1)), 1), n, replace = TRUE), ace)

output <- cbind(unique_ID, na, cl, k, urea, creat, glu, ast, alt, alb, Ca, aCa, Po4, crp, hb, mcv, neut, lymph, eosin, wcc, age, sex, bp, dmt1, dmt2, cvd, copd, obes, sglt2, ace)

# propensity score based outcome
mort_prop_score <- rep(1, n)
  ageScore = age ^ 3 / quantile(age^3)[3]
  mort_prop_score = mort_prop_score * ageScore
  
disease_vector <- c("bp", "dmt1", "dmt2", "cvd", "copd", "obes")
exp_rate <- c(1.2, 3, 1, 0.8, 0.7, 0.9)

for (i in seq(1, length(disease_vector), 1)) {
  disease_vector_values <- get(disease_vector[i])
  
  disease_prop_score <- rexp(n, rate = exp_rate[i])
  disease_prop_score <- ifelse(disease_prop_score < 1, 1, disease_prop_score)
  
  disease_prop <- rep(1, n)
  disease_prop <- ifelse(disease_vector_values == 1, disease_prop_score, disease_prop)
  
  mort_prop_score = mort_prop_score * disease_prop
}

labs_vector <- c("crp", "alb", "aCa", "hb", "lymph", "glu")
labs_thresholds <- c(20, 30, 2, 100, 4, 11)
labs_switch = c("higher", "lower",  "lower",  "lower", "higher", "higher")
exp_rate <- c(0.9, 0.7, 1, 0.8, 0.6, 1)

for (i in seq(1, length(labs_vector), 1)) {
  lab_vector_values <- get(labs_vector[i])
  
  lab_prop_score <- rexp(n, rate = exp_rate[i])
  lab_prop_score <- ifelse(lab_prop_score < 1, 1, lab_prop_score)
  
  lab_prop <- rep(1, n)
  if (labs_switch[i] == "higher") {
    lab_prop <- ifelse(lab_vector_values >= labs_thresholds[i], lab_prop_score, lab_prop)
  }
  if (labs_switch[i] == "lower") {
    lab_prop <- ifelse(lab_vector_values <= labs_thresholds[i], lab_prop_score, lab_prop)
  }

  mort_prop_score = mort_prop_score * lab_prop
}

# required mortality rate
mortality_rate <- 0.02
mort_thresh <- quantile(mort_prop_score, prob = c(1 - mortality_rate))
mort_thresh <- log(mort_thresh)

y <- rep(0, n)
y <- ifelse(log(mort_prop_score) > mort_thresh, 1, y)

output <- cbind(output, y)
output <- data.table(output)

write.table(output, file = "~/projects/COVID_models/r_syn_data.csv", sep = ",")
