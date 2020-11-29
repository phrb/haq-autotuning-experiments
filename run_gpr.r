library(randtoolbox)
library(dplyr)
library(tidyr)
library(rsm)
library(DiceKriging)
library(DiceOptim)
library(future.apply)

plan(multiprocess, workers = 40)

args = commandArgs(trailingOnly = TRUE)

quiet <- function(x) {
  sink(tempfile())
  on.exit(sink())
  invisible(force(x))
}

iterations <- 1

results <- NULL

# Resnet50
sobol_dim <- 54 * 1

# vgg19
# sobol_dim <- 19 * 2

starting_sobol_n <- (1 * sobol_dim) + 10
sobol_n <- starting_sobol_n

bit_min <- 1
bit_max <- 8
perturbation_range <- 2 * (bit_min / bit_max)

gpr_added_points <- 3
gpr_added_neighbours <- 3
gpr_neighbourhood_factor <- 500

EI_backlog <- NULL
EI_backlog_size <- 200

gpr_total_selected_points <- 1
gpr_iterations <- 245 - starting_sobol_n

gpr_sample_size <- 60 * sobol_dim

total_measurements <- starting_sobol_n + (gpr_iterations * gpr_total_selected_points)

network <- "resnet50"
network_sizes_data <- "network_sizes_data.csv"

preserve_ratio <- 0.1
batch_size <- 128
cuda_device <- as.integer(args[1])
resume_run_id <- as.integer(args[2])
resume_run_path <- args[3]

print(paste("Args:", args))
print(paste("Restoring from chosen path:",
            resume_run_path))

network_sizes <- read.csv(network_sizes_data)
network_specs <- network_sizes %>%
    filter(id == network)

design <- NULL
gpr_model <- NULL
df_design <- NULL
current_results <- NULL
size_df <- NULL
coded_size_df <- NULL
new_sample <- NULL
perturbation <- NULL
gpr_sample <- NULL
search_space <- NULL

# size_ratio: \in [0.0, 1.0], typical \in [0.1, 0.2]
# top1: \in [0.0, 100.0], typical \in [50, 95]
# top5: \in [0.0, 100.0], typical \in [65, 95]

size_weight <- 0.0
top1_weight <- 0.0
top5_weight <- 1.0

min_ratio <- 0.06

weights <- read.csv("resnet50_sizes.csv", header = TRUE)

sobol_partial <- 900000
sobol_neighbourhood_partial <- 500000
perturbed_sample_multiplier <- ceiling(sobol_neighbourhood_partial / gpr_added_points)

size_limits <- c(10.0, 1.0)

timing_info <- NULL

compute_size <- function(n, sample){
    as.numeric(weights[1,]) %*% (trunc(8 * as.numeric(sample[n, ])) + 1)
}

row_compute_size <- function(sample){
    as.numeric(weights[1,]) %*% (trunc(8 * as.numeric(sample)) + 1)
}

generate_filtered_sample <- function(size, sobol_n, limits){
    temp_sobol <- sobol(n = sobol_n,
                        dim = sobol_dim,
                        scrambling = 2,
                        seed = as.integer((99999 - 10000) * runif(1) + 10000),
                        init = TRUE)

    sobol_size <- sobol_n
    samples <- NULL
    filtered_samples = 0

    while(filtered_samples < size){
        print(paste("Current filtered samples:",
                    filtered_samples))
        design <- sobol(n = sobol_size,
                        dim = sobol_dim,
                        scrambling = 2,
                        seed = as.integer((99999 - 10000) * runif(1) + 10000),
                        init = FALSE)

        print("Generated design")

        sobol_size <- sobol_size * 2

        # Sequential apply
        # sizes <- sapply(1:length(design[,1]), compute_size, design)

        sizes <- future_apply(design, 1, row_compute_size)

        print("Applied filter")
        selected <- ((sizes / 8e6) < limits[1] & (sizes / 8e6) > limits[2])

        print("Selected valid samples")

        samples <- data.frame(design[selected, ])
        filtered_samples = length(samples[, 1])
        rm(design)
    }

    return(sample_n(samples, size))
}

perturb_filtered_sample <- function(sample, size, sobol_n, range, limits){
    temp_sobol <- sobol(n = sobol_n,
                        dim = sobol_dim,
                        scrambling = 2,
                        seed = as.integer((99999 - 10000) * runif(1) + 10000),
                        init = TRUE)

    sobol_size <- sobol_n
    samples <- NULL
    filtered_samples <- 0

    while(filtered_samples < size){
        print(paste("Current filtered samples:",
                    filtered_samples))
        perturbation <- sobol(n = sobol_size,
                              dim = sobol_dim,
                              scrambling = 2,
                              seed = as.integer((99999 - 10000) * runif(1) + 10000),
                              init = FALSE)

        print("Generated design")

        sobol_size <- sobol_size * 2

        perturbation <- data.frame(perturbation)
        perturbation <- (2 * range * perturbation) - range
        perturbed <- sample + perturbation

        perturbed[perturbed < 0.0] <- 0.1
        perturbed[perturbed > 1.0] <- 0.9

        sample <- bind_rows(sample, sample)

        # Sequential apply
        # sizes <- sapply(1:length(perturbed[, 1]), compute_size, perturbed)

        sizes <- future_apply(perturbed, 1, row_compute_size)

        print("Applied filter")
        selected <- ((sizes / 8e6) < limits[1] & (sizes / 8e6) > limits[2])

        samples <- data.frame(perturbed[selected, ])
        filtered_samples <- length(samples[, 1])

        rm(perturbed)
        rm(perturbation)
    }

    return(sample_n(samples, size))
}

performance <- function(size_ratio, top1, top5){
    return(((size_weight * (size_ratio - min_ratio) ^ 2) +
            (top1_weight * ((100.0 - top1) / 100.0)) +
            (top5_weight * ((100.0 - top5) / 100.0))) /
           (size_weight + top1_weight + top5_weight))
}

for(i in 1:iterations){
    if(!(is.null(gpr_sample))){
        rm(gpr_sample)
        quiet(gc())
        gpr_sample <- NULL
    }
    if(!(is.null(search_space))){
        rm(search_space)
        quiet(gc())
        search_space <- NULL
    }

    start_time <- as.integer(format(Sys.time(), "%s"))

    temp_sobol <- sobol(n = sobol_n,
                        dim = sobol_dim,
                        scrambling = 2,
                        seed = as.integer((99999 - 10000) * runif(1) + 10000),
                        init = TRUE)

    rm(temp_sobol)
    quiet(gc())


    if(i == 1 && resume_run_id != -1){
        print(paste("Resuming run:", resume_run_id))
        run_id <- resume_run_id
        initial_design_time <- -1
        initial_design_measure_time <- -1
    } else{
        run_id <- round(100000 * runif(1))

        if(!(is.null(design))){
            rm(design)
            quiet(gc())
            design <- NULL
        }

        initial_design_start_time <- proc.time()

        design <- generate_filtered_sample(sobol_n,
                                           sobol_partial,
                                           size_limits)

        initial_design_time <- (proc.time() - initial_design_start_time)[["elapsed"]]

        if(!(is.null(df_design))){
            rm(df_design)
            quiet(gc())
            df_design <- NULL
        }

        df_design <- design

        # names(df_design) <- c(rbind(paste("W",
        #                                   seq(1:(sobol_dim / 2)),
        #                                   sep = ""),
        #                             paste("A",
        #                                   seq(1:(sobol_dim / 2)),
        #                                   sep = "")))

        write.csv(df_design,
                  paste("current_design_",
                        run_id,
                        ".csv",
                        sep = ""),
                  row.names = FALSE)

        initial_design_measure_start_time <- proc.time()

        cmd <- paste("CUDA_VISIBLE_DEVICES=",
                     cuda_device,
                     " python3 -W ignore rl_quantize.py --arch ",
                     network,
                     " --dataset imagenet --dataset_root data",
                     " --suffix ratio010 --preserve_ratio ",
                     preserve_ratio,
                     " --n_worker 120 --warmup -1 --train_episode ",
                     sobol_n,
                     " --finetune_flag",
                     " --no-baseline",
                     " --use_top1",
                     " --run_id ",
                     run_id,
                     " --data_bsize ",
                     batch_size,
                     " --optimizer RS --val_size 10000",
                     " --train_size 20000",
                     sep = "")

        print(cmd)
        system(cmd)

        initial_design_measure_time <- (proc.time() -
                                        initial_design_measure_start_time)[["elapsed"]]

        system("rm -r ../../save")
    }

    if(!(is.null(current_results))){
        rm(current_results)
        quiet(gc())
        current_results <- NULL
    }

    if(!is.na(resume_run_path)){
        print(paste("Resuming run at path:", resume_run_path))
        current_results <- read.csv(resume_run_path,
                                    header = TRUE)
    } else{
        current_results <- read.csv(paste("current_results_",
                                          run_id,
                                          ".csv",
                                          sep = ""),
                                    header = TRUE)
    }

    if(is.null(search_space)){
        search_space <- current_results
    } else{
        search_space <- bind_rows(search_space, current_results) %>%
            distinct()
    }

    write.csv(search_space,
              paste("gpr_",
                    total_measurements,
                    "_samples_",
                    i,
                    "_iteration_id_",
                    run_id,
                    "_search_space.csv",
                    sep = ""),
              row.names = FALSE)

    for(j in 1:gpr_iterations){
        print("Starting reg")

        if(!(is.null(size_df))){
            rm(size_df)
            quiet(gc())
            size_df <- NULL
        }

        print("Search space:")
        print(str(search_space))

        if(!(is.null(gpr_model))){
            rm(gpr_model)
            quiet(gc())
            gpr_model <- NULL
        }

        y <- performance(search_space$SizeRatio, search_space$Top1, search_space$Top5)

        model_fit_start_time <- proc.time()

        gpr_model <- km(formula = ~ .,
                        design = select(search_space, -Top5, -Top1, -Size, -SizeRatio),
                        response = y,
                        nugget = 1e-8 * var(y),
                        control = list(pop.size = 400,
                                       BFGSburnin = 500))

        model_fit_time <- (proc.time() -
                           model_fit_start_time)[["elapsed"]]

        print("Generating Sample")

        ei_design_start_time <- proc.time()

        new_sample <- generate_filtered_sample(gpr_sample_size,
                                               sobol_partial,
                                               size_limits)

        ei_design_time <- (proc.time() -
                           ei_design_start_time)[["elapsed"]]

        gpr_sample <- new_sample %>%
            distinct()

        if(!is.null(EI_backlog)){
            gpr_sample <- bind_rows(gpr_sample,
                                    EI_backlog) %>%
                distinct()
        }

        rm(new_sample)
        quiet(gc())
        new_sample <- NULL

        print("Computing EI")
        print(nrow(gpr_sample))

        ei_compute_start_time <- proc.time()

        # Using the EI function from DiceOptim:
        gpr_sample$expected_improvement <- future_apply(gpr_sample,
                                                        1,
                                                        EI,
                                                        gpr_model)

        ei_compute_time <- (proc.time() -
                            ei_compute_start_time)[["elapsed"]]

        gpr_selected_points <- gpr_sample %>%
            arrange(desc(expected_improvement))

        # Using mean - 2 * sigma:
        # pred <- predict(gpr_model, gpr_sample, "UK")
        # gpr_sample$expected_improvement <- pred$mean - (1.96 * pred$sd)

        # gpr_selected_points <- gpr_sample %>%
        #     arrange(expected_improvement)

        gpr_sample <- select(gpr_sample, -expected_improvement)

        # gpr_selected_points <- select(gpr_selected_points[1:gpr_added_points, ],
        #                               -expected_improvement)

        gpr_selected_points <- select(gpr_selected_points[1:gpr_added_points, ],
                                      -expected_improvement)

        print("Generating perturbation sample")

        if(!(is.null(perturbation))){
            rm(perturbation)
            quiet(gc())
            perturbation <- NULL
        }

        perturbation <- gpr_selected_points %>%
            slice(rep(row_number(),
                      perturbed_sample_multiplier)) %>%
            slice(1:sobol_neighbourhood_partial)

        neighbour_ei_design_start_time <- proc.time()

        perturbation <- perturb_filtered_sample(perturbation,
                                                gpr_added_points * gpr_neighbourhood_factor,
                                                sobol_neighbourhood_partial,
                                                perturbation_range,
                                                size_limits)

        neighbour_ei_design_time <- (proc.time() -
                                     neighbour_ei_design_start_time)[["elapsed"]]

        gpr_selected_neighbourhood <- perturbation

        gpr_selected_points <- bind_rows(gpr_selected_points,
                                         gpr_selected_neighbourhood)

        gpr_selected_points <- gpr_selected_points %>%
            distinct()

        print("Computing perturbed EI")
        print(nrow(gpr_selected_points))

        neighbour_ei_compute_start_time <- proc.time()

        # Using EI from DiceOptim:
        gpr_selected_points$expected_improvement <- future_apply(gpr_selected_points,
                                                                 1,
                                                                 EI,
                                                                 gpr_model)

        neighbour_ei_compute_time <- (proc.time() -
                                      neighbour_ei_compute_start_time)[["elapsed"]]

        gpr_selected_points <- gpr_selected_points %>%
            arrange(desc(expected_improvement))

        # Using mean - 2 * sigma:
        # pred <- predict(gpr_model, gpr_selected_points, "UK")
        # gpr_selected_points$expected_improvement <- pred$mean - (1.96 * pred$sd)

        # gpr_selected_points <- gpr_selected_points %>%
        #     arrange(expected_improvement)

        # gpr_selected_points <- select(gpr_selected_points[1:(gpr_added_points +
        #                                                      gpr_added_neighbours), ],
        #                               -expected_improvement)

        EI_backlog <- select(gpr_selected_points[1:EI_backlog_size, ],
                             -expected_improvement)

        gpr_selected_points <- select(gpr_selected_points[1:gpr_total_selected_points, ],
                                      -expected_improvement)

        df_design <- data.frame(gpr_selected_points)

        write.csv(df_design,
                  paste("current_design_",
                        run_id,
                        ".csv",
                        sep = ""),
                  row.names = FALSE)

        neighbour_design_measure_start_time <- proc.time()

        cmd <- paste("CUDA_VISIBLE_DEVICES=",
                     cuda_device,
                     " python3 -W ignore rl_quantize.py --arch ",
                     network,
                     " --dataset imagenet --dataset_root data",
                     " --suffix ratio010 --preserve_ratio ",
                     preserve_ratio,
                     " --n_worker 120 --warmup -1 --train_episode ",
                     #gpr_added_points + gpr_added_neighbours,
                     gpr_total_selected_points,
                     " --finetune_flag",
                     " --no-baseline",
                     " --use_top1",
                     " --run_id ",
                     run_id,
                     " --data_bsize ",
                     batch_size,
                     " --optimizer RS --val_size 10000",
                     " --train_size 20000",
                     sep = "")

        print(cmd)
        system(cmd)

        neighbour_design_measure_time <- (proc.time() -
                                          neighbour_design_measure_start_time)[["elapsed"]]

        system("rm -r ../../save")

        if(!(is.null(current_results))){
            rm(current_results)
            quiet(gc())
            current_results <- NULL
        }

        current_results <- read.csv(paste("current_results_",
                                          run_id,
                                          ".csv",
                                          sep = ""),
                                    header = TRUE)

        if(is.null(search_space)){
            search_space <- current_results
        } else{
            search_space <- bind_rows(search_space, current_results) %>%
                distinct()
        }

        new_timing_info <- data.frame(initial_design = initial_design_time,
                                      initial_design_measure = initial_design_measure_time,
                                      model_fit = model_fit_time,
                                      ei_design = ei_design_time,
                                      ei_compute = ei_compute_time,
                                      neighbour_ei_design = neighbour_ei_design_time,
                                      neighbour_ei_compute = neighbour_ei_compute_time,
                                      neighbour_design_measure = neighbour_design_measure_time)

        if(is.null(timing_info)){
            timing_info <- new_timing_info
        } else{
            timing_info <- bind_rows(timing_info,
                                     new_timing_info)
        }

        write.csv(timing_info,
                  paste("timing_info_",
                        run_id,
                        ".csv",
                        sep = ""),
                  row.names = FALSE)

        write.csv(search_space,
                  paste("gpr_",
                        total_measurements,
                        "_samples_",
                        i,
                        "_iteration_id_",
                        run_id,
                        "_search_space.csv",
                        sep = ""),
                  row.names = FALSE)

        # if(length(search_space[, 1]) >= total_measurements){
        #     break
        # }
    }

    elapsed_time <- as.integer(format(Sys.time(), "%s")) - start_time

    response_data <- search_space %>%
        mutate(performance_metric = performance(search_space$SizeRatio,
                                                search_space$Top1,
                                                search_space$Top5))

    best_points <- search_space[response_data$performance_metric == min(response_data$performance_metric), ]

    best_points$id <- i
    best_points$elapsed_seconds <- elapsed_time
    best_points$points <- total_measurements

    best_points$gpr_iterations <- gpr_iterations
    best_points$gpr_added_points <- gpr_added_points
    best_points$perturbation_range <- perturbation_range
    best_points$gpr_neighbourhood <- gpr_neighbourhood_factor
    best_points$gpr_sample_size <- gpr_sample_size

    if(is.null(results)){
        results <- best_points
    } else{
        results <- bind_rows(results, best_points)
    }

    write.csv(results,
              paste("gpr_",
                    total_measurements,
                    "_samples_",
                    iterations,
                    "_iterations_id_",
                    run_id,
                    ".csv",
                    sep = ""),
              row.names = FALSE)
}
