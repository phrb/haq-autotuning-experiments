library(randtoolbox)
library(dplyr)
library(rsm)

quiet <- function(x) {
  sink(tempfile())
  on.exit(sink())
  invisible(force(x))
}

iterations <- 2

search_space <- NULL
results <- NULL

# Resnet50
sobol_dim <- 54 * 2

# VGG19
# sobol_dim <- 19 * 2

starting_sobol_n <- 2 * sobol_dim

sobol_n <- starting_sobol_n

bit_min <- 1
bit_max <- 8

for(i in 1:iterations){
    temp_sobol <- sobol(n = sobol_n,
                        dim = sobol_dim,
                        scrambling = 3,
                        seed = as.integer((99999 - 10000) * runif(1) + 10000),
                        init = TRUE)

    rm(temp_sobol)
    quiet(gc())

    design <- sobol(n = sobol_n,
                    dim = sobol_dim,
                    scrambling = 3,
                    seed = as.integer((99999 - 10000) * runif(1) + 10000),
                    init = FALSE)

    df_design <- data.frame(design)

    names(df_design) <- c(rbind(paste("W",
                                      seq(1:(sobol_dim / 2)),
                                      sep = ""),
                                paste("A",
                                      seq(1:(sobol_dim / 2)),
                                      sep = "")))

    coded_design <- df_design

    coded_df_design <- data.frame(coded_design)

    write.csv(coded_df_design, "current_design.csv", row.names = FALSE)

    start_time <- as.integer(format(Sys.time(), "%s"))

    cmd <- paste("CUDA_VISIBLE_DEVICES=0 python3 -W ignore rl_quantize.py",
                 " --arch ",
                 network,
                 " --dataset imagenet --dataset_root data",
                 " --suffix ratio010 --preserve_ratio 0.1 --n_worker 120",
                 " --warmup -1 --train_episode ",
                 sobol_n,
                 " --use_top5 --gpu_id 0",
                 " --data_bsize 128 --optimizer RS --val_size 10000 --train_size 20000",
                 sep = "")

    print(cmd)
    system(cmd)

    system("rm -r ../../save")

    elapsed_time <- as.integer(format(Sys.time(), "%s")) - start_time

    current_results <- read.csv("current_results.csv", header = TRUE)

    best_points <- filter(current_results, Top1 == max(Top1) | Top5 == max(Top5))
    best_points$id <- i
    best_points$elapsed_seconds <- elapsed_time
    best_points$points <- sobol_n

    if(is.null(results)){
        results <- best_points
    } else{
        results <- bind_rows(results, best_points)
    }

    if(is.null(search_space)){
        search_space <- current_results
    } else{
        search_space <- bind_rows(search_space, current_results)
    }

    write.csv(results,
              paste("rs_",
                    starting_sobol_n,
                    "_samples_",
                    iterations,
                    "_iterations.csv",
                    sep = ""),
              row.names = FALSE)


    write.csv(search_space,
              paste("rs_",
                    starting_sobol_n,
                    "_samples_",
                    iterations,
                    "_iterations_search_space.csv",
                    sep = ""),
              row.names = FALSE)
}
