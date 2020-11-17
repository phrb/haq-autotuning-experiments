library(dplyr)

args = commandArgs(trailingOnly = TRUE)

quiet <- function(x) {
  sink(tempfile())
  on.exit(sink())
  invisible(force(x))
}

iterations <- 1
results <- NULL


network <- "resnet50"
preserve_ratio <- 0.1
batch_size <- 128
cuda_device <- as.integer(args[1])
warmup <- 64
total_measurements <- 244

for(i in 1:iterations){
    run_id <- round(100000 * runif(1))

    search_space <- NULL
    start_time <- as.integer(format(Sys.time(), "%s"))

    cmd <- paste("CUDA_VISIBLE_DEVICES=",
                 cuda_device,
                 " python3 -W ignore rl_quantize.py --arch ",
                 network,
                 " --dataset imagenet --dataset_root data",
                 " --suffix ratio010 --preserve_ratio ",
                 preserve_ratio,
                 " --n_worker 120 --warmup ",
                 warmup,
                 " --train_episode ",
                 total_measurements,
                 " --finetune_flag",
                 " --use_top5",
                 " --run_id ",
                 run_id,
                 " --data_bsize ",
                 batch_size,
                 " --optimizer DDPG --val_size 10000",
                 " --train_size 20000",
                 sep = "")

    print(cmd)
    system(cmd)

    system("rm -r ../../save")

    elapsed_time <- as.integer(format(Sys.time(), "%s")) - start_time

    current_results <- read.csv(paste("haq_results_log_",
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
              paste("ddpg_",
                    total_measurements,
                    "_samples_",
                    i,
                    "_iteration_run_id_",
                    run_id,
                    "_search_space.csv",
                    sep = ""),
              row.names = FALSE)

    best_points <- filter(search_space, Top5 == max(Top5))
    best_points$id <- i
    best_points$elapsed_seconds <- elapsed_time
    best_points$points <- total_measurements

    if(is.null(results)){
        results <- best_points
    } else{
        results <- bind_rows(results, best_points)
    }

    write.csv(results,
              paste("ddpg_",
                    total_measurements,
                    "_samples_",
                    iterations,
                    "_iterations.csv",
                    sep = ""),
              row.names = FALSE)
}
