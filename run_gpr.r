library(randtoolbox)
library(dplyr)
library(tidyr)
library(rsm)
library(DiceKriging)
library(DiceOptim)
library(future.apply)

plan(multiprocess, workers = 256)

args = commandArgs(trailingOnly = TRUE)

quiet <- function(x) {
  sink(tempfile())
  on.exit(sink())
  invisible(force(x))
}

iterations <- 2

results <- NULL

# Resnet50
sobol_dim <- 54 * 2

# vgg19
# sobol_dim <- 19 * 2

starting_sobol_n <- (1 * sobol_dim) + 10
sobol_n <- starting_sobol_n

bit_min <- 1
bit_max <- 8
perturbation_range <- 4 * (bit_min / bit_max)

gpr_iterations <- 40
gpr_added_points <- 3

gpr_added_neighbours <- 2
gpr_neighbourhood_factor <- 1000

gpr_sample_size <- 50 * sobol_dim

total_measurements <- starting_sobol_n + (gpr_iterations * (gpr_added_points + gpr_added_neighbours))

network <- "resnet50"
network_sizes_data <- "network_sizes_data.csv"

preserve_ratio <- 0.1
batch_size <- 128
cuda_device <- as.integer(args[1])
resume_run_id <- as.integer(args[2])

size_weight <- 1
top1_weight <- 0
top5_weight <- 0

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
    } else{
        run_id <- round(100000 * runif(1))

        if(!(is.null(design))){
            rm(design)
            quiet(gc())
            design <- NULL
        }

        design <- sobol(n = sobol_n,
                        dim = sobol_dim,
                        scrambling = 2,
                        seed = as.integer((99999 - 10000) * runif(1) + 10000),
                        init = FALSE)

        if(!(is.null(df_design))){
            rm(df_design)
            quiet(gc())
            df_design <- NULL
        }

        df_design <- data.frame(design)

        names(df_design) <- c(rbind(paste("W",
                                          seq(1:(sobol_dim / 2)),
                                          sep = ""),
                                    paste("A",
                                          seq(1:(sobol_dim / 2)),
                                          sep = "")))

        write.csv(df_design,
                  paste("current_design_",
                        run_id,
                        ".csv",
                        sep = ""),
                  row.names = FALSE)

        cmd <- paste("CUDA_VISIBLE_DEVICES=",
                     cuda_device,
                     " python3 -W ignore rl_quantize.py --arch ",
                     network,
                     " --dataset imagenet --dataset_root data",
                     " --suffix ratio010 --preserve_ratio ",
                     preserve_ratio,
                     " --n_worker 120 --warmup -1 --train_episode ",
                     sobol_n,
                     " --use_top5",
                     " --run_id ",
                     run_id,
                     " --data_bsize ",
                     batch_size,
                     " --optimizer RS --val_size 10000",
                     " --train_size 20000",
                     sep = "")

        print(cmd)
        system(cmd)

        system("rm -r ../../save")
    }

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

        size_df <- select(search_space, -Top5, -Top1)
        formulas <- character(length(names(size_df)))

        for(k in 1:length(names(size_df))){
            formulas[k] <- paste(names(size_df)[k],
                                 "e ~ round((7 * ",
                                 names(size_df)[k],
                                 ") + 1)",
                                 sep = "")
        }

        if(!(is.null(coded_size_df))){
            rm(coded_size_df)
            quiet(gc())
            coded_size_df <- NULL
        }

        coded_size_df <- coded.data(size_df, formulas = lapply(formulas, formula))
        coded_size_df <- round(data.frame(coded_size_df))
        coded_size_df$id <- seq(1:length(coded_size_df$A1e))
        coded_size_df <- gather(coded_size_df, "Layer", "Bitwidth", -id)

        coded_size_df <- coded_size_df %>%
            group_by(id) %>%
            do(mutate(., weights_MB = sum((network_specs$parameters *
                                           (filter(., grepl("W", Layer))$Bitwidth / 8)) / 1e6))) %>%
            do(mutate(., activations_MB = sum((network_specs$activations *
                                               (filter(., grepl("A", Layer))$Bitwidth / 8)) / 1e6))) %>%
            summarize(total_size_MB = unique(weights_MB) + unique(activations_MB),
                      network_size_MB = sum(network_specs$bits8_size_MB))

        print("Search space:")
        print(str(search_space))
        print("Coded weight df:")
        print(str(coded_size_df))

        if(!(is.null(gpr_model))){
            rm(gpr_model)
            quiet(gc())
            gpr_model <- NULL
        }

        y <- ((size_weight * (coded_size_df$total_size_MB / coded_size_df$network_size_MB)) +
              (top1_weight * ((100.0 - search_space$Top1) / 100.0)) +
              (top5_weight * ((100.0 - search_space$Top5) / 100.0))) /
            (size_weight + top1_weight + top5_weight)

        gpr_model <- km(formula = ~ .,
                        design = select(search_space, -Top5, -Top1),
                        response = y,
                        nugget = 1e-8 * var(y),
                        control = list(pop.size = 400,
                                       BFGSburnin = 500))

        print("Generating Sample")

        if(!(is.null(new_sample))){
            rm(new_sample)
            quiet(gc())
            new_sample <- NULL
        }

        new_sample <- sobol(n = gpr_sample_size,
                            dim = sobol_dim,
                            scrambling = 2,
                            seed = as.integer((99999 - 10000) * runif(1) + 10000),
                            init = FALSE)

        new_sample <- data.frame(new_sample)

        names(new_sample) <- c(rbind(paste("W",
                                           seq(1:(sobol_dim / 2)),
                                           sep = ""),
                                     paste("A",
                                           seq(1:(sobol_dim / 2)),
                                           sep = "")))

        if(is.null(gpr_sample)){
            gpr_sample <- new_sample %>%
                distinct()
        } else{
            gpr_sample <- bind_rows(gpr_sample, new_sample) %>%
                distinct()
        }

        print("Computing EI")
        # gpr_sample$expected_improvement <- future_apply(gpr_sample,
        #                                                 1,
        #                                                 EI,
        #                                                 gpr_model)
        pred <- predict(gpr_model, gpr_sample, "UK")
        gpr_sample$expected_improvement <- pred$mean - (1.96 * pred$sd)

        gpr_selected_points <- gpr_sample %>%
            arrange(desc(expected_improvement))

        gpr_sample <- select(gpr_sample, -expected_improvement)

        gpr_selected_points <- select(gpr_selected_points[1:gpr_added_points, ],
                                      -expected_improvement)

        print("Generating perturbation sample")

        if(!(is.null(perturbation))){
            rm(perturbation)
            quiet(gc())
            perturbation <- NULL
        }

        perturbation <- sobol(n = gpr_added_points * gpr_neighbourhood_factor,
                              dim = sobol_dim,
                              scrambling = 2,
                              seed = as.integer((99999 - 10000) * runif(1) + 10000),
                              init = FALSE)

        perturbation <- data.frame(perturbation)

        names(perturbation) <- c(rbind(paste("W",
                                            seq(1:(sobol_dim / 2)),
                                            sep = ""),
                                      paste("A",
                                            seq(1:(sobol_dim / 2)),
                                            sep = "")))

        perturbation <- (2 * perturbation_range * perturbation) - perturbation_range

        gpr_selected_neighbourhood <- gpr_selected_points %>%
            slice(rep(row_number(), gpr_neighbourhood_factor))

        gpr_selected_neighbourhood <- gpr_selected_neighbourhood + perturbation

        gpr_selected_neighbourhood[gpr_selected_neighbourhood < 0.0] <- 0.124
        gpr_selected_neighbourhood[gpr_selected_neighbourhood > 1.0] <- 0.876

        gpr_sample <- bind_rows(gpr_sample, gpr_selected_neighbourhood) %>%
            distinct()

        gpr_selected_points <- bind_rows(gpr_selected_points,
                                         gpr_selected_neighbourhood)
        gpr_selected_points <- gpr_selected_points %>%
            distinct()

        print("Computing perturbed EI")
        # gpr_selected_points$expected_improvement <- future_apply(gpr_selected_points,
        #                                                          1,
        #                                                          EI,
        #                                                          gpr_model)

        pred <- predict(gpr_model, gpr_selected_points, "UK")
        gpr_selected_points$expected_improvement <- pred$mean - (1.96 * pred$sd)

        gpr_selected_points <- gpr_selected_points %>%
            arrange(desc(expected_improvement))

        gpr_selected_points <- select(gpr_selected_points[1:(gpr_added_points +
                                                             gpr_added_neighbours), ],
                                      -expected_improvement)

        df_design <- data.frame(gpr_selected_points)
        names(df_design) <- c(rbind(paste("W",
                                          seq(1:(sobol_dim / 2)),
                                          sep = ""),
                                    paste("A",
                                          seq(1:(sobol_dim / 2)),
                                          sep = "")))

        write.csv(df_design,
                  paste("current_design_",
                        run_id,
                        ".csv",
                        sep = ""),
                  row.names = FALSE)

        cmd <- paste("CUDA_VISIBLE_DEVICES=",
                     cuda_device,
                     " python3 -W ignore rl_quantize.py --arch ",
                     network,
                     " --dataset imagenet --dataset_root data",
                     " --suffix ratio010 --preserve_ratio ",
                     preserve_ratio,
                     " --n_worker 120 --warmup -1 --train_episode ",
                     gpr_added_points + gpr_added_neighbours,
                     " --use_top5",
                     " --run_id ",
                     run_id,
                     " --data_bsize ",
                     batch_size,
                     " --optimizer RS --val_size 10000",
                     " --train_size 20000",
                     sep = "")

        print(cmd)
        system(cmd)

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
    }

    elapsed_time <- as.integer(format(Sys.time(), "%s")) - start_time

    # Optimizing for Top5 (Top1, Model Size, Latency... ?)

    # top5 only
    # best_points <- filter(search_space, Top5 == max(Top5))

    size_df <- select(search_space, -Top5, -Top1)
    formulas <- character(length(names(size_df)))

    for(k in 1:length(names(size_df))){
        formulas[k] <- paste(names(size_df)[k],
                             "e ~ round((7 * ",
                             names(size_df)[k],
                             ") + 1)",
                             sep = "")
    }

    coded_size_df <- coded.data(size_df, formulas = lapply(formulas, formula))
    coded_size_df <- round(data.frame(coded_size_df))
    coded_size_df$id <- seq(1:length(coded_size_df$A1e))
    coded_size_df <- gather(coded_size_df, "Layer", "Bitwidth", -id)

    coded_size_df <- coded_size_df %>%
        group_by(id) %>%
        do(mutate(., weights_MB = sum((network_specs$parameters *
                                       (filter(., grepl("W", Layer))$Bitwidth / 8)) / 1e6))) %>%
        do(mutate(., activations_MB = sum((network_specs$activations *
                                           (filter(., grepl("A", Layer))$Bitwidth / 8)) / 1e6))) %>%
        summarize(total_size_MB = unique(weights_MB) + unique(activations_MB),
                  network_size_MB = sum(network_specs$bits8_size_MB))

    response_data <- search_space %>%
        mutate(performance_metric = ((size_weight * (coded_size_df$total_size_MB / coded_size_df$network_size_MB)) +
                                     (top1_weight * ((100.0 - search_space$Top1) / 100.0)) +
                                     (top5_weight * ((100.0 - search_space$Top5) / 100.0))) /
                   (size_weight + top1_weight + top5_weight))

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
