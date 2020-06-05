install.packages("reticulate")
library(reticulate)

py_run_string("from simulation import SimulationLeadBraking")
py_run_string("s = SimulationLeadBraking()")

my_func <- function(theta){
    # theta=(initial speed, average deceleration, speed difference)
    py_run_string(sprintf("prob = s.get_probability((%f, %f, %f), seed=0)",
                          theta[1], theta[2], theta[3]))
    py$prob
}

# Idea 1:
# Fix initial speed at 20 m/s
# Vary average deceleration between 0.5 m/s^2 till 5.0 m/s^2
# Vary speed difference between 5 m/s till 20 m/s

# Idea 2:
# Vary initial speed between 5 m/s and 40 m/s
# Vary average deceleration between 0.5 m/s^2 till 5.0 m/s^2
# Vary speed difference between 5 m/s till the initial speed
