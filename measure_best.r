library(dplyr)

args = commandArgs(trailingOnly = TRUE)

read_search_space_csv <- function(path){
    df = read.csv(path)
    df$experiment_id = path
    return(df)
}

read_data <- function(new_path, pattern = "search_space.csv"){
    csv_files = list.files(path = new_path,
                            pattern = pattern)
    target = paste0(new_path, csv_files)

    return(bind_rows(lapply(target, read_search_space_csv)))
}

load_best_points <- function(target_path){
    return(df = read_data(target_path) %>%
               group_by(experiment_id) %>%
               filter(n() >= 245 & Top5 == max(Top5)) %>%
               ungroup())
}

run_measurement <- function(measurement, cuda_device){
    runs = 1

    preserve_ratio = 0.1
    batch_size = 128
    network = "resnet50"

    run_id = round(100000 * runif(1))

    print("Writing sample")

    write.csv(measurement %>%
              select(-Top1, -Top5, -experiment_id, -Size, -SizeRatio),
              paste("current_design_",
                    run_id,
                    ".csv",
                    sep = ""),
              row.names = FALSE)

    cmd = paste("CUDA_VISIBLE_DEVICES=",
                cuda_device,
                " python3 -W ignore rl_quantize.py --arch ",
                network,
                " --dataset imagenet --dataset_root data",
                " --suffix ratio010 --preserve_ratio ",
                preserve_ratio,
                " --n_worker 120 --warmup -1 --train_episode ",
                runs,
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

    print(paste("Current run:", measurement$experiment_id))
    print(paste("Running command:", cmd))

    system(cmd)
    system("rm -r ../../save")

    current_results = read.csv(paste("current_results_",
                                     run_id,
                                     ".csv",
                                     sep = ""),
                               header = TRUE)

    system(paste("rm current_design_",
                 run_id,
                 ".csv",
                 sep = ""))

    system(paste("rm current_results_",
                 run_id,
                 ".csv",
                 sep = ""))

    return(c(current_results$Top1[[1]],
             current_results$Top5[[1]]))
}

repetitions = 10
target_path = "results/resnet50_tests/gpr_restricted_top15_1experiment/"
cuda_device = as.integer(args[1])

df = load_best_points(target_path)
measurements = NULL

for(i in 1:repetitions){
    results = sapply(1:nrow(df),
                     function (x) { run_measurement(df[x, ], cuda_device) })

    new_measurements = df
    new_measurements$Top1_repeats = results[1]
    new_measurements$Top5_repeats = results[2]

    if(is.null(measurements)){
        measurements = new_measurements
    } else{
        measurements = bind_rows(measurements,
                                 new_measurements)
    }

    write.csv(measurements,
              "measure_best.csv",
              row.names = FALSE)
}
