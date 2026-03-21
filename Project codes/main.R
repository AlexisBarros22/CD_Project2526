# Function to check if libraries are installed and install them if not
check_install_packages <- function(packages) {
  installed_packages <- rownames(installed.packages())
  missing_packages <- packages[!(packages %in% installed_packages)]

  if (length(missing_packages) > 0) {
    cat("Installing missing packages:", paste(missing_packages, collapse = ", "), "\n")
    install.packages(missing_packages)
  } else {
    cat("All required packages are already installed.\n")
  }
}

# List of required packages
required_packages <- c("methods", "ggplot2", "pracma", "patchwork", "gridExtra", "car", "coin", "stats", "utils", "datasets", "dplyr", "uwot")  # Add more if needed

# Check and install missing packages
check_install_packages(required_packages)
library(dplyr)
library(R6)
library(ggplot2)
library(gridExtra)
library(pracma)
library(car)
library(coin)
library(stats)
library(patchwork)
library(datasets)
library(methods)
library(uwot)

DataLoader <- R6Class("DataLoader",
  public = list(
    data = NULL,

    initialize = function() {
      self$data <- private$load_data()
      self$show_data()
    },

    # Equivalente a show_data(self)
    show_data = function() {
      if (!is.null(self$data)) {
        cat("--- Data Preview ---\n")
        # Equivalente ao df.head()
        print(head(self$data))

        cat("\n--- Data Summary ---\n")
        # Equivalente ao df.describe()
        print(summary(self$data))
      } else {
        cat("No data loaded\n")
      }
    }
  ),

  private = list(
    load_data = function() {
      result <- tryCatch({

        file_path <- "Project Datasets/flights_sample_3m.csv"

        df <- read.csv(file_path, fileEncoding = "latin1")
        return(df)

      }, error = function(e) {
        cat("Error loading data: ", conditionMessage(e), "\n")
        return(NULL)
      })

      return(result)
    }
  )
)

DataPreprocess <- R6Class("DataPreprocess",
  public = list(
    data = NULL,
    verbose = NULL,

    # Equivalent to __init__
    initialize = function(data, verbose = TRUE) {
      # Use copy to prevent modifying the original dataframe by reference
      self$data <- as.data.frame(data)
      self$verbose <- verbose
    },

    drop_columns = function() {
      columns_to_drop <- c(
        'DEP_DELAY', 'DELAY_DUE_CARRIER', 'DELAY_DUE_WEATHER', 'DELAY_DUE_SECURITY',
        'DELAY_DUE_NAS', 'DELAY_DUE_LATE_AIRCRAFT', 'ARR_TIME', 'DEP_TIME',
        'WHEELS_OFF', 'WHEELS_ON', 'TAXI_OUT', 'TAXI_IN', 'ELAPSED_TIME',
        'AIR_TIME', 'CANCELLATION_CODE', 'AIRLINE', 'AIRLINE_CODE',
        'AIRLINE_DOT', 'FL_NUMBER', 'ORIGIN', 'DEST'
      )

      # any_of prevents errors if the column doesn't exist, like errors="ignore"
      self$data <- self$data %>% select(-any_of(columns_to_drop))

      if (self$verbose) {
        print(head(self$data))
      }

      invisible(self)
    },

    report_missing_values = function() {
      if (!self$verbose) return(invisible(self))

      cat("Total flights:", nrow(self$data), "\n\n")
      cat("NA values per column:\n")
      print(colSums(is.na(self$data)))

      na_rows <- sum(!complete.cases(self$data))
      cat("\nTotal rows with at least one NA value:", na_rows, "\n")
      cat(sprintf("Percentage of rows with NA: %.2f%%\n", (na_rows / nrow(self$data) * 100)))

      if ("CANCELLED" %in% names(self$data)) {
        cat("\nCancelled flights:", sum(self$data$CANCELLED, na.rm = TRUE), "\n")
      }
      if ("DIVERTED" %in% names(self$data)) {
        cat("Diverted flights:", sum(self$data$DIVERTED, na.rm = TRUE), "\n")
      }

      invisible(self)
    },

    filter_cancelled_diverted = function() {
      if (all(c('CANCELLED', 'DIVERTED') %in% names(self$data))) {
        self$data <- self$data %>%
          filter(CANCELLED == 0 & DIVERTED == 0) %>%
          select(-any_of(c('CANCELLED', 'DIVERTED')))
      }

      if (self$verbose) {
        cat("\nTotal flights after filtering:", nrow(self$data), "\n")
      }

      invisible(self)
    },

    clean_na = function() {
      if (self$verbose) {
        cat("\nNA values before dropping:\n")
        print(colSums(is.na(self$data)))
      }

      # complete.cases is base R's equivalent to dropna()
      self$data <- self$data[complete.cases(self$data), ]

      if (self$verbose) {
        cat("\nTotal flights after dropping NA:", nrow(self$data), "\n")
      }

      invisible(self)
    },

    add_date_features = function() {
      if ("FL_DATE" %in% names(self$data)) {
        # Convert to Date object if not already
        if (!inherits(self$data$FL_DATE, "Date")) {
          self$data$FL_DATE <- as.Date(self$data$FL_DATE)
        }

        # Extract Year, Month, and Day of Week (1-7)
        self$data <- self$data %>%
          mutate(
            FL_YEAR = as.numeric(format(FL_DATE, "%Y")),
            FL_MONTH = as.numeric(format(FL_DATE, "%m")),
            FL_DAY_OF_WEEK = as.numeric(format(FL_DATE, "%u"))
          )

        if (self$verbose) {
          cat("\nDate features extracted:\n")
          print(head(self$data %>% select(FL_DATE, FL_YEAR, FL_MONTH, FL_DAY_OF_WEEK)))
        }

        self$data <- self$data %>% select(-FL_DATE)
      }

      invisible(self)
    },

    convert_scheduled_times_cyclical = function() {
      time_cols <- c('CRS_DEP_TIME', 'CRS_ARR_TIME')

      for (col in time_cols) {
        if (col %in% names(self$data)) {
          self$data[[col]] <- as.numeric(self$data[[col]])

          hours <- self$data[[col]] %/% 100
          minutes <- self$data[[col]] %% 100
          total_minutes <- (hours * 60) + minutes

          minutes_in_day <- 24 * 60

          # Create cyclical features
          sin_col <- paste0(col, "_sin")
          cos_col <- paste0(col, "_cos")

          self$data[[sin_col]] <- sin(2 * pi * total_minutes / minutes_in_day)
          self$data[[cos_col]] <- cos(2 * pi * total_minutes / minutes_in_day)

          # Drop the original column to match Python behavior
          self$data <- self$data %>% select(-all_of(col))

          if (self$verbose) {
            cat(sprintf("\nConverted %s to cyclical features:\n", col))
            print(head(self$data %>% select(all_of(c(sin_col, cos_col)))))
          }
        }
      }

      invisible(self)
    },

    convert_to_season = function() {
      if ("FL_MONTH" %in% names(self$data) && !"SEASON" %in% names(self$data)) {
        self$data <- self$data %>%
          mutate(SEASON = case_when(
            FL_MONTH %in% c(12, 1, 2) ~ 1,
            FL_MONTH %in% c(3, 4, 5)  ~ 2,
            FL_MONTH %in% c(6, 7, 8)  ~ 3,
            FL_MONTH %in% c(9, 10, 11) ~ 4,
            TRUE ~ NA_real_
          ))

        if (self$verbose) {
          cat("\nSeason feature created:\n")
          print(head(self$data %>% select(FL_MONTH, SEASON)))
        }
      }

      invisible(self)
    },

    is_weekend = function() {
      if ("FL_DAY_OF_WEEK" %in% names(self$data) && !"IS_WEEKEND" %in% names(self$data)) {
        self$data <- self$data %>%
          mutate(IS_WEEKEND = ifelse(FL_DAY_OF_WEEK %in% c(6, 7), 1, 0))

        if (self$verbose) {
          cat("\nWeekend feature created:\n")
          print(head(self$data %>% select(FL_DAY_OF_WEEK, IS_WEEKEND)))
        }
      }

      invisible(self)
    },

    route = function() {
      if (all(c('ORIGIN_CITY', 'DEST_CITY') %in% names(self$data)) && !"ROUTE" %in% names(self$data)) {
        self$data <- self$data %>%
          mutate(ROUTE = paste(ORIGIN_CITY, DEST_CITY, sep = "_"))

        if (self$verbose) {
          cat("\nRoute feature created:\n")
          print(head(self$data %>% select(ORIGIN_CITY, DEST_CITY, ROUTE)))
        }
      }

      invisible(self)
    },

    avg_speed = function() {
      if (all(c('DISTANCE', 'CRS_ELAPSED_TIME') %in% names(self$data)) && !"AVG_SPEED" %in% names(self$data)) {
        self$data <- self$data %>%
          mutate(AVG_SPEED = DISTANCE / CRS_ELAPSED_TIME)

        if (self$verbose) {
          cat("\nAverage speed feature created:\n")
          print(head(self$data %>% select(DISTANCE, CRS_ELAPSED_TIME, AVG_SPEED)))
        }
      }

      invisible(self)
    },

    dep_hour = function() {
      if ("CRS_DEP_TIME" %in% names(self$data) && !"DEP_HOUR" %in% names(self$data)) {
        self$data <- self$data %>% mutate(DEP_HOUR = CRS_DEP_TIME %/% 60)

        if (self$verbose) {
          cat("\nDeparture hour feature created:\n")
          print(head(self$data %>% select(CRS_DEP_TIME, DEP_HOUR)))
        }
      }
      invisible(self)
    },

    arr_hour = function() {
      if ("CRS_ARR_TIME" %in% names(self$data) && !"ARR_HOUR" %in% names(self$data)) {
        self$data <- self$data %>% mutate(ARR_HOUR = CRS_ARR_TIME %/% 60)

        if (self$verbose) {
          cat("\nArrival hour feature created:\n")
          print(head(self$data %>% select(CRS_ARR_TIME, ARR_HOUR)))
        }
      }
      invisible(self)
    },

    peak_morning = function() {
      if ("DEP_HOUR" %in% names(self$data) && !"PEAK_MORNING" %in% names(self$data)) {
        self$data <- self$data %>%
          mutate(PEAK_MORNING = ifelse(DEP_HOUR >= 7 & DEP_HOUR <= 10, 1, 0))

        if (self$verbose) {
          cat("\nMorning peak feature created:\n")
          print(head(self$data %>% select(DEP_HOUR, PEAK_MORNING)))
        }
      }
      invisible(self)
    },

    peak_evening = function() {
      if ("DEP_HOUR" %in% names(self$data) && !"PEAK_EVENING" %in% names(self$data)) {
        self$data <- self$data %>%
          mutate(PEAK_EVENING = ifelse(DEP_HOUR >= 16 & DEP_HOUR <= 19, 1, 0))

        if (self$verbose) {
          cat("\nEvening peak feature created:\n")
          print(head(self$data %>% select(DEP_HOUR, PEAK_EVENING)))
        }
      }
      invisible(self)
    },

    origin_state = function() {
      if ("ORIGIN_CITY" %in% names(self$data) && !"ORIGIN_STATE" %in% names(self$data)) {
        # Uses regex to extract everything after the last comma and trim whitespace
        self$data <- self$data %>%
          mutate(ORIGIN_STATE = trimws(sub(".*,", "", as.character(ORIGIN_CITY))))

        if (self$verbose) {
          cat("\nOrigin state feature created:\n")
          print(head(self$data %>% select(ORIGIN_CITY, ORIGIN_STATE)))
        }
      }
      invisible(self)
    },

    dest_state = function() {
      if ("DEST_CITY" %in% names(self$data) && !"DEST_STATE" %in% names(self$data)) {
        self$data <- self$data %>%
          mutate(DEST_STATE = trimws(sub(".*,", "", as.character(DEST_CITY))))

        if (self$verbose) {
          cat("\nDestination state feature created:\n")
          print(head(self$data %>% select(DEST_CITY, DEST_STATE)))
        }
      }
      invisible(self)
    },

    fix_negative_delays = function() {
      cols_to_check <- c("ARR_DELAY")

      for (col in cols_to_check) {
        if (col %in% names(self$data)) {

          negative_count <- sum(self$data[[col]] < 0, na.rm = TRUE)

          if (negative_count > 0) {
            if (self$verbose) {
              cat(sprintf("\nFound %d negative values in '%s'. Setting them to 0.\n", negative_count, col))
            }

            self$data[[col]][self$data[[col]] < 0 & !is.na(self$data[[col]])] <- 0
          }
        }
      }

      invisible(self)
    },

    export_to_csv = function(path) {
      write.csv(self$data, file = path, row.names = FALSE)
      if (self$verbose) {
        cat(sprintf("\nData exported to: %s\n", path))
      }
      invisible(self)
    },

    get_data = function() {
      return(self$data)
    }
  )
)

DataSplit <- R6Class("DataSplit",
  public = list(
    data = NULL,
    test_size = NULL,
    random_state = NULL,
    verbose = NULL,

    data_train_eda = NULL,
    data_test_eda = NULL,
    data_train = NULL,
    labels_train = NULL,
    data_test = NULL,
    labels_test = NULL,

    categorical_cols = c('ORIGIN_CITY', 'DEST_CITY', 'ORIGIN_STATE', 'DEST_STATE', 'ROUTE'),
    normalize_columns = c('CRS_DEP_TIME', 'CRS_ARR_TIME', 'DISTANCE', 'CRS_ELAPSED_TIME', 'AVG_SPEED', 'DEP_HOUR', 'ARR_HOUR'),

    state_mapping = NULL,
    route_mapping = NULL,
    other_mappings = list(),
    scaler_center = NULL,
    scaler_scale = NULL,

    initialize = function(data, test_size = 0.2, random_state = 48, verbose = TRUE) {
      self$data <- as.data.frame(data)
      self$test_size <- test_size
      self$random_state <- random_state
      self$verbose <- verbose

      private$split_data()
      private$encode_categorical()
      private$scale_numeric()
    },

    export_encoding_mappings = function(path) {
      rows <- list()

      # 1) Shared state mapping
      if (!is.null(self$state_mapping) && length(self$state_mapping) > 0) {
        for (state in names(self$state_mapping)) {
          rows[[length(rows) + 1]] <- data.frame(
            Column = 'STATE_SHARED',
            Original_Value = state,
            Encoded_Code = self$state_mapping[[state]],
            stringsAsFactors = FALSE
          )
        }
      }

      # 2) Route mapping
      if (!is.null(self$route_mapping) && length(self$route_mapping) > 0) {
        for (route in names(self$route_mapping)) {
          rows[[length(rows) + 1]] <- data.frame(
            Column = 'ROUTE',
            Original_Value = route,
            Encoded_Code = self$route_mapping[[route]],
            stringsAsFactors = FALSE
          )
        }
      }

      # 3) Remaining OrdinalEncoder mappings
      if (length(self$other_mappings) > 0) {
        for (col in names(self$other_mappings)) {
          mapping <- self$other_mappings[[col]]
          for (orig_val in names(mapping)) {
            rows[[length(rows) + 1]] <- data.frame(
              Column = col,
              Original_Value = orig_val,
              Encoded_Code = mapping[[orig_val]],
              stringsAsFactors = FALSE
            )
          }
        }
      }

      if (length(rows) == 0) {
        if (self$verbose) cat("No encoding mappings to export.\n")
        return(invisible(self))
      }

      mappings_df <- bind_rows(rows)
      write.csv(mappings_df, file = path, row.names = FALSE)

      if (self$verbose) {
        cat(sprintf("\nEncoding mappings exported to: %s\n", path))
        print(head(mappings_df, 10))
      }

      invisible(self)
    }
  ),

  private = list(
    split_data = function() {
      # Use base R subsetting here to avoid NSE issues inside R6
      X <- self$data[, !(names(self$data) %in% c("ARR_DELAY")), drop = FALSE]
      y <- self$data$ARR_DELAY

      set.seed(self$random_state)
      n_rows <- nrow(X)

      # Using sample.int is slightly faster and safer than sample() for indices
      train_indices <- sample.int(n_rows, size = floor((1 - self$test_size) * n_rows))

      self$data_train_eda <- X[train_indices, , drop = FALSE]
      self$data_test_eda  <- X[-train_indices, , drop = FALSE]

      self$data_train <- self$data_train_eda
      self$data_test  <- self$data_test_eda
      self$labels_train <- y[train_indices]
      self$labels_test  <- y[-train_indices]
    },

    encode_categorical = function() {
      # ---------- 1) Shared mapping for states ----------
      state_cols <- intersect(c('ORIGIN_STATE', 'DEST_STATE'), names(self$data_train))

      if (length(state_cols) > 0) {
        # Faster way to get all unique states
        all_states <- sort(unique(unlist(lapply(self$data_train[state_cols], as.character))))

        self$state_mapping <- setNames(seq_along(all_states) - 1, all_states)

        for (col in state_cols) {
          self$data_train[[col]] <- unname(self$state_mapping[as.character(self$data_train[[col]])])
          self$data_train[[col]][is.na(self$data_train[[col]])] <- -1

          self$data_test[[col]] <- unname(self$state_mapping[as.character(self$data_test[[col]])])
          self$data_test[[col]][is.na(self$data_test[[col]])] <- -1
        }
      }

      # ---------- 2) Symmetric encoding for route ----------
      if ('ROUTE' %in% names(self$data_train)) {
        # Vectorized string split and sort for massive speedup on 2.9M rows
        canonical_route <- function(routes) {
          # vapply is safer and faster than sapply here
          vapply(strsplit(as.character(routes), "_"), function(x) {
            if(length(x) != 2) return(paste(x, collapse="_"))
            paste(sort(x), collapse="_")
          }, FUN.VALUE = character(1))
        }

        train_routes <- canonical_route(self$data_train$ROUTE)
        test_routes  <- canonical_route(self$data_test$ROUTE)

        unique_routes <- sort(unique(train_routes))
        self$route_mapping <- setNames(seq_along(unique_routes) - 1, unique_routes)

        self$data_train$ROUTE <- unname(self$route_mapping[train_routes])
        self$data_train$ROUTE[is.na(self$data_train$ROUTE)] <- -1

        self$data_test$ROUTE <- unname(self$route_mapping[test_routes])
        self$data_test$ROUTE[is.na(self$data_test$ROUTE)] <- -1
      }

      # ---------- 3) Separate encoding for remaining categorical columns ----------
      remaining_cols <- setdiff(self$categorical_cols, c('ORIGIN_STATE', 'DEST_STATE', 'ROUTE'))
      remaining_cols <- intersect(remaining_cols, names(self$data_train))

      if (length(remaining_cols) > 0) {
        self$other_mappings <- list()

        for (col in remaining_cols) {
          unique_vals <- sort(unique(as.character(self$data_train[[col]])))
          mapping <- setNames(seq_along(unique_vals) - 1, unique_vals)
          self$other_mappings[[col]] <- mapping

          self$data_train[[col]] <- unname(mapping[as.character(self$data_train[[col]])])
          self$data_train[[col]][is.na(self$data_train[[col]])] <- -1

          self$data_test[[col]] <- unname(mapping[as.character(self$data_test[[col]])])
          self$data_test[[col]][is.na(self$data_test[[col]])] <- -1
        }
      }
    },

    scale_numeric = function() {
      cols_present <- intersect(self$normalize_columns, names(self$data_train))

      if (length(cols_present) == 0) return()

      scaled_train <- scale(self$data_train[cols_present])
      self$scaler_center <- attr(scaled_train, "scaled:center")
      self$scaler_scale <- attr(scaled_train, "scaled:scale")

      self$data_train[cols_present] <- as.data.frame(scaled_train)

      scaled_test <- scale(self$data_test[cols_present], center = self$scaler_center, scale = self$scaler_scale)
      self$data_test[cols_present] <- as.data.frame(scaled_test)

      if (self$verbose) {
        cat(sprintf("\nData split successful: %d training samples, %d testing samples.\n",
                    nrow(self$data_train), nrow(self$data_test)))
        cat("Scaled columns:", paste(cols_present, collapse = ", "), "\n")
      }
    }
  )
)

EDA <- R6Class("EDA",
  public = list(
    data = NULL,
    verbose = NULL,
    base_theme = NULL,

    initialize = function(data, verbose = TRUE) {
      self$data <- as.data.frame(data)
      self$verbose <- verbose

      # Global plot style mimicking seaborn whitegrid/talk context
      self$base_theme <- theme_minimal(base_size = 14) +
        theme(
          plot.background = element_rect(fill = "white", color = NA),
          panel.background = element_rect(fill = "#f8f9fa", color = "#333333"),
          panel.grid.major = element_line(color = "grey90"),
          panel.grid.minor = element_blank(),
          plot.title = element_text(face = "bold", hjust = 0.5)
        )
    },

    summary = function() {
      if (!self$verbose) return(invisible(self))

      cat("Exploratory Data Analysis (EDA) Report\n")
      cat(strrep("-", 50), "\n")
      cat(sprintf("Shape: %d rows, %d columns\n", nrow(self$data), ncol(self$data)))
      cat("\nColumns:\n")
      print(names(self$data))
      cat("\nData types:\n")
      print(sapply(self$data, class))
      cat("\nMissing values:\n")
      print(colSums(is.na(self$data)))
      cat("\nSummary statistics:\n")
      print(summary(self$data))

      invisible(self)
    },

    plot_target_distribution = function(bins = 80, clip_range = c(-60, 180), export = FALSE) {
      if (!"ARR_DELAY" %in% names(self$data)) {
        cat("ARR_DELAY column not found.\n")
        return(invisible(self))
      }

      plot_data <- data.frame(
        ARR_DELAY = pmax(clip_range[1], pmin(clip_range[2], self$data$ARR_DELAY))
      )

      p <- ggplot(plot_data, aes(x = ARR_DELAY)) +
        geom_histogram(aes(y = after_stat(density)), bins = bins, fill = "dodgerblue", color = "white", alpha = 0.85) +
        geom_density(color = "#005b96", linewidth = 1) +
        labs(title = "Arrival Delay Distribution (Clipped)", x = "Arrival Delay (minutes)", y = "Density") +
        self$base_theme

      print(p)
      if (export) private$export_current_plot(p, "distributions", "arrival_delay_distribution_clipped")
      invisible(self)
    },

    plot_numeric_distributions = function(columns = 2, bins = 40, export = FALSE) {
      if (is.null(columns)) {
        columns <- c("CRS_ELAPSED_TIME", "DISTANCE", "AVG_SPEED", "ARR_DELAY")
      }
      existing_cols <- intersect(columns, names(self$data))

      if (length(existing_cols) == 0) {
        cat("No valid numeric columns found.\n")
        return(invisible(self))
      }

      plots <- list()
      for (col in existing_cols) {
        plot_vals <- self$data[[col]]
        if (col == 'ARR_DELAY') {
          plot_vals <- pmax(-60, pmin(180, plot_vals))
        }

        df <- data.frame(val = plot_vals)
        plots[[col]] <- ggplot(df, aes(x = val)) +
          geom_histogram(aes(y = after_stat(density)), bins = bins, fill = "#66c2a5", color = "white", alpha = 0.9) +
          geom_density(color = "#3288bd", linewidth = 0.8) +
          labs(title = sprintf("Distribution of %s", col), x = col, y = "Density") +
          self$base_theme
      }

      combined_plot <- wrap_plots(plots, ncol = 2)
      print(combined_plot)
      if (export) private$export_current_plot(combined_plot, "distributions", "numeric_distributions", width=14, height=5*ceiling(length(plots)/2))
      invisible(self)
    },

    plot_boxplots = function(columns = 3, clip_dict = NULL, export = FALSE) {
      if (is.null(columns)) {
        columns <- c("CRS_ELAPSED_TIME", "DISTANCE", "ARR_DELAY", "AVG_SPEED")
      }

      if (is.null(clip_dict)) {
        clip_dict <- list(
          'CRS_ELAPSED_TIME' = c(0, 400), 'DISTANCE' = c(0, 3000),
          'ARR_DELAY' = c(-60, 300), 'AVG_SPEED' = c(1, 6)
        )
      }

      existing_cols <- intersect(columns, names(self$data))
      if (length(existing_cols) == 0) return(invisible(self))

      plots <- list()
      for (col in existing_cols) {
        plot_vals <- self$data[[col]]
        if (col %in% names(clip_dict)) {
          limits <- clip_dict[[col]]
          plot_vals <- pmax(limits[1], pmin(limits[2], plot_vals))
        }

        df <- data.frame(val = plot_vals)
        plots[[col]] <- ggplot(df, aes(x = val)) +
          geom_boxplot(fill = "#fdae61", color = "#d53e4f", width = 0.5, outlier.shape = NA) +
          labs(title = sprintf("Boxplot of %s (Clipped)", col), x = col) +
          self$base_theme +
          theme(axis.text.y = element_blank(), axis.ticks.y = element_blank())
      }

      combined_plot <- wrap_plots(plots, ncol = 1)
      print(combined_plot)
      if (export) private$export_current_plot(combined_plot, "boxplots", "continuous_boxplots", width=12, height=3.5*length(plots))
      invisible(self)
    },

    plot_cyclical_time_features = function(export = FALSE) {
      pairs <- list(
        c("CRS_DEP_TIME_sin", "CRS_DEP_TIME_cos", "Scheduled Departure Time"),
        c("CRS_ARR_TIME_sin", "CRS_ARR_TIME_cos", "Scheduled Arrival Time")
      )

      plots <- list()
      for (pair in pairs) {
        if (all(pair[1:2] %in% names(self$data))) {
          df <- self$data[, pair[1:2]]
          df <- df[complete.cases(df), ]
          names(df) <- c("sin", "cos")

          plots[[pair[3]]] <- ggplot(df, aes(x = cos, y = sin)) +
            geom_point(alpha = 0.25, size = 1) +
            annotate("path", x=cos(seq(0,2*pi,length.out=100)), y=sin(seq(0,2*pi,length.out=100)), linetype="dashed", alpha=0.5) +
            coord_fixed(xlim = c(-1.1, 1.1), ylim = c(-1.1, 1.1)) +
            labs(title = pair[3], x = "Cosine", y = "Sine") +
            self$base_theme
        }
      }

      if(length(plots) > 0) {
        combined_plot <- wrap_plots(plots, nrow = 1)
        print(combined_plot)
        if (export) private$export_current_plot(combined_plot, "cyclical_features", "cyclical_time_features", width=7*length(plots), height=6)
      }
      invisible(self)
    },

    plot_correlation_heatmap = function(export = FALSE) {
      target_cols <- c("DISTANCE", "CRS_ELAPSED_TIME", "ARR_DELAY", "AVG_SPEED",
                       "PEAK_MORNING", "PEAK_EVENING", "CRS_DEP_TIME_sin",
                       "CRS_DEP_TIME_cos", "CRS_ARR_TIME_sin", "CRS_ARR_TIME_cos")
      cols_present <- intersect(target_cols, names(self$data))

      if (length(cols_present) < 2) return(invisible(self))

      corr_matrix <- cor(self$data[, cols_present], use = "pairwise.complete.obs")
      corr_df <- as.data.frame(as.table(corr_matrix))

      p <- ggplot(corr_df, aes(x = Var1, y = Var2, fill = Freq)) +
        geom_tile(color = "white") +
        geom_text(aes(label = sprintf("%.2f", Freq)), size = 3) +
        scale_fill_gradient2(low = "#4575b4", mid = "white", high = "#d73027", midpoint = 0, limit = c(-1, 1), name = "Corr") +
        labs(title = "Correlation Heatmap", x = "", y = "") +
        self$base_theme +
        theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1))

      print(p)
      if (export) private$export_current_plot(p, "correlations", "correlation_heatmap", width=12, height=8)
      invisible(self)
    },

    plot_delay_by_day_of_week = function(export = FALSE) {
      if (!all(c('FL_DAY_OF_WEEK', 'ARR_DELAY') %in% names(self$data))) return(invisible(self))

      delay_by_day <- self$data %>%
        group_by(FL_DAY_OF_WEEK) %>%
        summarize(ARR_DELAY = mean(ARR_DELAY, na.rm = TRUE), .groups = 'drop')

      p <- ggplot(delay_by_day, aes(x = as.factor(FL_DAY_OF_WEEK), y = ARR_DELAY, fill = as.factor(FL_DAY_OF_WEEK))) +
        geom_col(show.legend = FALSE) +
        scale_fill_viridis_d() +
        labs(title = "Mean Arrival Delay by Day of Week", x = "Day of Week", y = "Mean Delay (minutes)") +
        self$base_theme

      print(p)
      if (export) private$export_current_plot(p, "barplots", "mean_arrival_delay_by_day_of_week", width=10, height=5)
      invisible(self)
    },

    plot_delay_rate_by_day_of_week = function(export = FALSE) {
      if (!all(c('FL_DAY_OF_WEEK', 'ARR_DELAY') %in% names(self$data))) return(invisible(self))

      delay_rate <- self$data %>%
        mutate(DELAYED = ifelse(ARR_DELAY > 0, 1, 0)) %>%
        group_by(FL_DAY_OF_WEEK) %>%
        summarize(DELAYED = mean(DELAYED, na.rm = TRUE) * 100, .groups = 'drop')

      p <- ggplot(delay_rate, aes(x = as.factor(FL_DAY_OF_WEEK), y = DELAYED, fill = as.factor(FL_DAY_OF_WEEK))) +
        geom_col(show.legend = FALSE) +
        scale_fill_viridis_d(option = "magma") +
        labs(title = "Proportion of Delayed Flights by Day of Week", x = "Day of Week", y = "Delay Rate (%)") +
        self$base_theme

      print(p)
      if (export) private$export_current_plot(p, "barplots", "delay_rate_by_day_of_week", width=10, height=5)
      invisible(self)
    },

    plot_delay_by_month = function(export = FALSE) {
      if (!all(c('FL_MONTH', 'ARR_DELAY') %in% names(self$data))) return(invisible(self))

      delay_by_month <- self$data %>%
        group_by(FL_MONTH) %>%
        summarize(ARR_DELAY = mean(ARR_DELAY, na.rm = TRUE), .groups = 'drop')

      p <- ggplot(delay_by_month, aes(x = FL_MONTH, y = ARR_DELAY)) +
        geom_line(color = "#008080", linewidth = 1.5) +
        geom_point(color = "darkslategray", size = 3) +
        scale_x_continuous(breaks = 1:12) +
        labs(title = "Mean Arrival Delay by Month", x = "Month", y = "Mean Delay (minutes)") +
        self$base_theme

      print(p)
      if (export) private$export_current_plot(p, "lineplots", "mean_arrival_delay_by_month", width=12, height=5)
      invisible(self)
    },

    plot_delay_vs_distance = function(export = FALSE) {
      if (!all(c('DISTANCE', 'ARR_DELAY') %in% names(self$data))) return(invisible(self))

      df <- data.frame(
        DISTANCE = self$data$DISTANCE,
        ARR_DELAY = pmax(-60, pmin(180, self$data$ARR_DELAY))
      )

      p <- ggplot(df, aes(x = DISTANCE, y = ARR_DELAY)) +
        geom_point(alpha = 0.35, size = 1, color = "steelblue") +
        geom_smooth(method = "lm", color = "red", linewidth = 1) +
        labs(title = "Arrival Delay vs Distance", x = "Distance (miles)", y = "Arrival Delay (minutes, clipped)") +
        self$base_theme

      print(p)
      if (export) private$export_current_plot(p, "scatterplots", "arrival_delay_vs_distance", width=10, height=6)
      invisible(self)
    },

    plot_delay_vs_elapsed_time = function(export = FALSE) {
      if (!all(c('CRS_ELAPSED_TIME', 'ARR_DELAY') %in% names(self$data))) return(invisible(self))

      df <- data.frame(
        CRS_ELAPSED_TIME = self$data$CRS_ELAPSED_TIME,
        ARR_DELAY = pmax(-60, pmin(180, self$data$ARR_DELAY))
      )

      p <- ggplot(df, aes(x = CRS_ELAPSED_TIME, y = ARR_DELAY)) +
        geom_point(alpha = 0.35, size = 1, color = "steelblue") +
        geom_smooth(method = "lm", color = "red", linewidth = 1) +
        labs(title = "Arrival Delay vs Scheduled Elapsed Time", x = "Scheduled Elapsed Time (minutes)", y = "Arrival Delay (minutes, clipped)") +
        self$base_theme

      print(p)
      if (export) private$export_current_plot(p, "scatterplots", "arrival_delay_vs_elapsed_time", width=10, height=6)
      invisible(self)
    },

    plot_delay_heatmap_month_day = function(export = FALSE) {
      if (!all(c('FL_MONTH', 'FL_DAY_OF_WEEK', 'ARR_DELAY') %in% names(self$data))) return(invisible(self))

      pivot <- self$data %>%
        group_by(FL_DAY_OF_WEEK, FL_MONTH) %>%
        summarize(ARR_DELAY = mean(ARR_DELAY, na.rm = TRUE), .groups = 'drop')

      p <- ggplot(pivot, aes(x = as.factor(FL_MONTH), y = as.factor(FL_DAY_OF_WEEK), fill = ARR_DELAY)) +
        geom_tile(color = "white") +
        geom_text(aes(label = sprintf("%.1f", ARR_DELAY)), color = "black", size = 3) +
        scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = mean(pivot$ARR_DELAY)) +
        labs(title = "Mean Arrival Delay by Month and Day of Week", x = "Month", y = "Day of Week") +
        self$base_theme

      print(p)
      if (export) private$export_current_plot(p, "heatmaps", "mean_arrival_delay_month_day", width=12, height=6)
      invisible(self)
    },

    plot_departure_time_month_heatmap = function(bins = 24, export = FALSE) {
      req_cols <- c("CRS_DEP_TIME_sin", "CRS_DEP_TIME_cos", "FL_MONTH", "ARR_DELAY")
      if (!all(req_cols %in% names(self$data))) return(invisible(self))

      df_plot <- self$data[, req_cols]
      df_plot <- df_plot[complete.cases(df_plot), ]

      angles <- atan2(df_plot$CRS_DEP_TIME_sin, df_plot$CRS_DEP_TIME_cos)
      angles <- (angles + 2 * pi) %% (2 * pi)

      breaks <- seq(0, 2 * pi, length.out = bins + 1)
      labels <- sprintf("%02d:00", 0:(bins-1))
      df_plot$time_bin <- cut(angles, breaks = breaks, labels = labels, include.lowest = TRUE)

      pivot <- df_plot %>%
        group_by(time_bin, FL_MONTH) %>%
        summarize(ARR_DELAY = mean(ARR_DELAY, na.rm=TRUE), .groups='drop')

      p <- ggplot(pivot, aes(x = as.factor(FL_MONTH), y = time_bin, fill = ARR_DELAY)) +
        geom_tile() +
        scale_fill_viridis_c(option="viridis") +
        labs(title = "Mean Delay by Departure Time and Month", x = "Month", y = "Departure Time Bin") +
        self$base_theme

      print(p)
      if (export) private$export_current_plot(p, "heatmaps", "mean_delay_departure_time_month", width=12, height=8)
      invisible(self)
    },

    plot_delay_by_departure_time_circle = function(bins = 24, export = FALSE) {
      req_cols <- c("CRS_DEP_TIME_sin", "CRS_DEP_TIME_cos", "ARR_DELAY")
      if (!all(req_cols %in% names(self$data))) return(invisible(self))

      df_plot <- self$data[, req_cols]
      df_plot <- df_plot[complete.cases(df_plot), ]

      angles <- atan2(df_plot$CRS_DEP_TIME_sin, df_plot$CRS_DEP_TIME_cos)
      angles <- (angles + 2 * pi) %% (2 * pi)

      breaks <- seq(0, 2 * pi, length.out = bins + 1)
      df_plot$time_bin <- cut(angles, breaks = breaks, labels = FALSE, include.lowest = TRUE)

      summary <- df_plot %>%
        group_by(time_bin) %>%
        summarize(ARR_DELAY = mean(ARR_DELAY, na.rm=TRUE), .groups='drop')

      # Simple polar approximation in ggplot
      p <- ggplot(summary, aes(x = time_bin, y = ARR_DELAY)) +
        geom_polygon(fill = "dodgerblue", alpha = 0.25, color = "blue") +
        geom_point() +
        coord_polar(theta = "x") +
        labs(title = "Mean Arrival Delay by Scheduled Departure Time") +
        theme_minimal()

      print(p)
      if (export) private$export_current_plot(p, "polar_plots", "mean_arrival_delay_by_departure_time", width=8, height=8)
      invisible(self)
    },

    plot_route_delay_rate = function(top_n = 15, min_flights = 3000, export = FALSE) {
      if (!all(c('ROUTE', 'ARR_DELAY') %in% names(self$data))) return(invisible(self))

      stats <- self$data %>%
        mutate(DELAYED = ifelse(ARR_DELAY > 0, 1, 0)) %>%
        group_by(ROUTE) %>%
        summarize(delay_rate = mean(DELAYED, na.rm=TRUE) * 100, flights = n(), .groups='drop') %>%
        filter(flights >= min_flights) %>%
        arrange(desc(flights)) %>%
        head(top_n)

      p <- ggplot(stats, aes(x = delay_rate, y = reorder(ROUTE, delay_rate), fill = delay_rate)) +
        geom_col(show.legend = FALSE) +
        scale_fill_viridis_c() +
        labs(title = "Delay Rate (%) for Top Busy Routes", x = "Delay Rate (%)", y = "Route") +
        self$base_theme

      print(p)
      if (export) private$export_current_plot(p, "barplots", "route_delay_rate_busy_routes", width=12, height=6)
      invisible(self)
    },

    plot_delay_by_season_violin = function(export = FALSE) {
      if (!all(c('SEASON', 'ARR_DELAY') %in% names(self$data))) return(invisible(self))

      df_plot <- data.frame(
        SEASON = as.factor(self$data$SEASON),
        ARR_DELAY = pmax(-60, pmin(180, self$data$ARR_DELAY))
      )

      p <- ggplot(df_plot, aes(x = SEASON, y = ARR_DELAY, fill = SEASON)) +
        geom_violin(trim = FALSE, alpha = 0.7) +
        geom_boxplot(width = 0.1, fill = "white", outlier.shape = NA) +
        labs(title = "Arrival Delay Distribution by Season", x = "Season", y = "Arrival Delay (minutes, clipped)") +
        self$base_theme + theme(legend.position = "none")

      print(p)
      if (export) private$export_current_plot(p, "violin_plots", "arrival_delay_by_season", width=10, height=6)
      invisible(self)
    },

    plot_origin_city_volume_vs_delay = function(min_flights = 2000, export = FALSE) {
      if (!all(c('ORIGIN_CITY', 'ARR_DELAY') %in% names(self$data))) return(invisible(self))

      stats <- self$data %>%
        group_by(ORIGIN_CITY) %>%
        summarize(avg_delay = mean(ARR_DELAY, na.rm=TRUE), flights = n(), .groups='drop') %>%
        filter(flights >= min_flights)

      p <- ggplot(stats, aes(x = flights, y = avg_delay)) +
        geom_point(alpha = 0.7, color = "darkred", size=2) +
        labs(title = "Origin City: Flight Volume vs Average Delay", x = "Number of Flights", y = "Average Arrival Delay") +
        self$base_theme

      print(p)
      if (export) private$export_current_plot(p, "scatterplots", "origin_city_volume_vs_average_delay", width=12, height=7)
      invisible(self)
    },

    plot_all_core = function() {
      self$summary()$
        plot_target_distribution()$
        plot_numeric_distributions()$
        plot_boxplots()$
        plot_cyclical_time_features()$
        plot_correlation_heatmap()$
        plot_delay_by_day_of_week()$
        plot_delay_rate_by_day_of_week()$
        plot_delay_by_month()$
        plot_delay_vs_distance()$
        plot_delay_vs_elapsed_time()$
        plot_delay_heatmap_month_day()$
        plot_departure_time_month_heatmap()$
        plot_delay_by_departure_time_circle()$
        plot_route_delay_rate()$
        plot_delay_by_season_violin()$
        plot_origin_city_volume_vs_delay()

      invisible(self)
    },

    export_all_core = function() {
      self$plot_target_distribution(export=TRUE)$
        plot_numeric_distributions(export=TRUE)$
        plot_boxplots(export=TRUE)$
        plot_cyclical_time_features(export=TRUE)$
        plot_correlation_heatmap(export=TRUE)$
        plot_delay_by_day_of_week(export=TRUE)$
        plot_delay_rate_by_day_of_week(export=TRUE)$
        plot_delay_by_month(export=TRUE)$
        plot_delay_vs_distance(export=TRUE)$
        plot_delay_vs_elapsed_time(export=TRUE)$
        plot_delay_heatmap_month_day(export=TRUE)$
        plot_departure_time_month_heatmap(export=TRUE)$
        plot_delay_by_departure_time_circle(export=TRUE)$
        plot_route_delay_rate(export=TRUE)$
        plot_delay_by_season_violin(export=TRUE)$
        plot_origin_city_volume_vs_delay(export=TRUE)

      invisible(self)
    }
  ),

  private = list(
    sanitize_filename = function(name) {
      safe <- gsub("[^[:alnum:]_\\- ]", "_", name)
      safe <- trimws(safe)
      safe <- gsub(" ", "_", safe)
      return(tolower(safe))
    },

    ensure_plot_folder = function(plot_type) {
      base_dir <- file.path("..", "Output files")
      plot_dir <- file.path(base_dir, plot_type)
      dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)
      return(plot_dir)
    },

    export_current_plot = function(p, plot_type, plot_name, dpi = 300, width=10, height=6) {
      folder <- private$ensure_plot_folder(plot_type)
      filename <- paste0(private$sanitize_filename(plot_name), ".png")
      filepath <- file.path(folder, filename)

      ggsave(filename = filepath, plot = p, dpi = dpi, width = width, height = height, bg="white")

      if (self$verbose) {
        cat(sprintf("Saved plot to: %s\n", filepath))
      }
    }
  )
)

DimReduction <- R6Class("DimReduction",
  public = list(
    data = NULL,
    labels = NULL,
    verbose = NULL,

    pca_result = NULL,
    pca_model = NULL,
    pca_labels = NULL,

    umap_result = NULL,
    umap_labels = NULL,
    base_theme = NULL,

    initialize = function(data, labels, verbose = TRUE) {
      self$data <- as.data.frame(data)
      self$labels <- as.vector(labels)
      self$verbose <- verbose

      # Global plot style mimicking seaborn whitegrid
      self$base_theme <- theme_minimal(base_size = 14) +
        theme(
          plot.background = element_rect(fill = "white", color = NA),
          panel.background = element_rect(fill = "#f8f9fa", color = "#333333"),
          panel.grid.major = element_line(color = "grey90"),
          panel.grid.minor = element_blank(),
          plot.title = element_text(face = "bold", hjust = 0.5)
        )
    },

    run_pca = function(feature_cols = NULL) {
      if (self$verbose) cat("\n--- Running PCA (Linear) ---\n")

      cols_present <- private$get_feature_columns(feature_cols)

      if (length(cols_present) < 2) {
        cat("Not enough feature columns found to run PCA.\n")
        return(invisible(self))
      }

      numeric_data <- self$data[, cols_present, drop = FALSE]

      # Note: sklearn's PCA centers by default but does NOT scale.
      # To match exactly, we set center = TRUE and scale. = FALSE
      self$pca_model <- prcomp(numeric_data, center = TRUE, scale. = FALSE)
      self$pca_result <- self$pca_model$x
      self$pca_labels <- self$labels

      if (self$verbose) {
        var_explained <- self$pca_model$sdev^2 / sum(self$pca_model$sdev^2)
        cat(sprintf("Features used: %s\n", paste(cols_present, collapse = ", ")))
        cat(sprintf("PC1 explains %.2f%% of the variance.\n", var_explained[1] * 100))
        if (length(var_explained) > 1) {
          cat(sprintf("PC2 explains %.2f%% of the variance.\n", var_explained[2] * 100))
        }
      }

      invisible(self)
    },

    plot_pca = function(max_samples = 100000, label_mode = "delay_categorical", export = FALSE) {
      if (is.null(self$pca_result)) {
        cat("Please run_pca() first.\n")
        return(invisible(self))
      }

      n_rows <- nrow(self$pca_result)

      if (n_rows > max_samples) {
        if (self$verbose) cat(sprintf("Plotting a random sample of %s points for visibility...\n", format(max_samples, big.mark = ",")))
        set.seed(42)
        idx <- sample.int(n_rows, max_samples)
      } else {
        idx <- seq_len(n_rows)
      }

      plot_df <- data.frame(
        PC1 = self$pca_result[idx, 1],
        PC2 = self$pca_result[idx, 2]
      )

      # Handle coloring based on label_mode
      if (label_mode == "delay_categorical") {
        plot_df$Label <- private$categorize_delay_labels(self$pca_labels[idx])
        plot_df$Label <- factor(plot_df$Label, levels = c("On-time / Early", "Minor delay", "Moderate delay", "Long delay"))

        palette <- c("On-time / Early" = "#4575b4", "Minor delay" = "#91bfdb",
                     "Moderate delay" = "#fdae61", "Long delay" = "#d73027")

        p <- ggplot(plot_df, aes(x = PC1, y = PC2, color = Label)) +
          geom_point(alpha = 0.5, size = 1, stroke = 0) +
          scale_color_manual(values = palette) +
          theme(legend.position = "bottom", legend.title = element_blank())

      } else if (label_mode == "generic_categorical") {
        plot_df$Label <- as.factor(private$generic_categorical_labels(self$pca_labels[idx]))

        p <- ggplot(plot_df, aes(x = PC1, y = PC2, color = Label)) +
          geom_point(alpha = 0.5, size = 1, stroke = 0) +
          labs(color = "Category")

      } else if (label_mode == "continuous") {
        plot_df$Label <- self$pca_labels[idx]

        p <- ggplot(plot_df, aes(x = PC1, y = PC2, color = Label)) +
          geom_point(alpha = 0.5, size = 1, stroke = 0) +
          scale_color_distiller(palette = "RdYlBu", direction = -1, name = "Value") # Closest to coolwarm

      } else {
        cat("label_mode must be 'delay_categorical', 'generic_categorical', or 'continuous'.\n")
        return(invisible(self))
      }

      # Add shared titles and themes
      p <- p +
        labs(
          title = sprintf("PCA: Flight Data Patterns (n=%s)", format(length(idx), big.mark = ",")),
          x = "Principal Component 1",
          y = "Principal Component 2"
        ) +
        self$base_theme

      print(p)
      if (export) private$export_current_plot(p, "dimensionality_reduction", paste0("pca_", label_mode))
      invisible(self)
    },

    run_umap = function(feature_cols = NULL, n_neighbors = 15, min_dist = 0.1, max_samples = 100000) {
      if (self$verbose) cat("\n--- Running UMAP (Non-linear) ---\n")

      cols_present <- private$get_feature_columns(feature_cols)

      if (length(cols_present) < 2) {
        cat("Not enough feature columns found to run UMAP.\n")
        return(invisible(self))
      }

      if (nrow(self$data) > max_samples) {
        if (self$verbose) cat(sprintf("Dataset too large. Downsampling to %s rows...\n", format(max_samples, big.mark = ",")))
        set.seed(42)
        idx <- sample.int(nrow(self$data), max_samples)

        numeric_data <- self$data[idx, cols_present, drop = FALSE]
        self$umap_labels <- self$labels[idx]
      } else {
        numeric_data <- self$data[, cols_present, drop = FALSE]
        self$umap_labels <- self$labels
      }

      if (self$verbose) {
        cat(sprintf("Features used: %s\n", paste(cols_present, collapse = ", ")))
        cat("Calculating UMAP...\n")
      }

      # Run UMAP using the uwot package
      set.seed(42)
      self$umap_result <- uwot::umap(
        numeric_data,
        n_neighbors = n_neighbors,
        min_dist = min_dist,
        init = "random",
        verbose = FALSE
      )

      if (self$verbose) cat("UMAP complete.\n")

      invisible(self)
    },

    plot_umap = function(label_mode = "delay_categorical", export = FALSE) {
      if (is.null(self$umap_result)) {
        cat("Please run_umap() first.\n")
        return(invisible(self))
      }

      plot_df <- data.frame(
        UMAP1 = self$umap_result[, 1],
        UMAP2 = self$umap_result[, 2]
      )

      if (label_mode == "delay_categorical") {
        plot_df$Label <- private$categorize_delay_labels(self$umap_labels)
        plot_df$Label <- factor(plot_df$Label, levels = c("On-time / Early", "Minor delay", "Moderate delay", "Long delay"))

        palette <- c("On-time / Early" = "#4575b4", "Minor delay" = "#91bfdb",
                     "Moderate delay" = "#fdae61", "Long delay" = "#d73027")

        p <- ggplot(plot_df, aes(x = UMAP1, y = UMAP2, color = Label)) +
          geom_point(alpha = 0.5, size = 1, stroke = 0) +
          scale_color_manual(values = palette) +
          theme(legend.position = "bottom", legend.title = element_blank())

      } else if (label_mode == "generic_categorical") {
        plot_df$Label <- as.factor(private$generic_categorical_labels(self$umap_labels))

        p <- ggplot(plot_df, aes(x = UMAP1, y = UMAP2, color = Label)) +
          geom_point(alpha = 0.5, size = 1, stroke = 0) +
          labs(color = "Category")

      } else if (label_mode == "continuous") {
        plot_df$Label <- self$umap_labels

        p <- ggplot(plot_df, aes(x = UMAP1, y = UMAP2, color = Label)) +
          geom_point(alpha = 0.5, size = 1, stroke = 0) +
          scale_color_distiller(palette = "RdYlBu", direction = -1, name = "Value")

      } else {
        cat("label_mode must be 'delay_categorical', 'generic_categorical', or 'continuous'.\n")
        return(invisible(self))
      }

      p <- p +
        labs(title = "UMAP: Flight Data Patterns", x = "UMAP Dimension 1", y = "UMAP Dimension 2") +
        self$base_theme

      print(p)
      if (export) private$export_current_plot(p, "dimensionality_reduction", paste0("umap_", label_mode))
      invisible(self)
    },

    plot_all_core = function() {
      self$run_pca()$
        plot_pca()$
        run_umap()$
        plot_umap()

      invisible(self)
    },

    export_all_core = function() {
      self$run_pca()$
        plot_pca(label_mode = "delay_categorical", export = TRUE)$
        plot_pca(label_mode = "generic_categorical", export = TRUE)$
        plot_pca(label_mode = "continuous", export = TRUE)$
        run_umap()$
        plot_umap(label_mode = "delay_categorical", export = TRUE)$
        plot_umap(label_mode = "generic_categorical", export = TRUE)$
        plot_umap(label_mode = "continuous", export = TRUE)

      invisible(self)
    }
  ),

  private = list(
    sanitize_filename = function(name) {
      safe <- gsub("[^[:alnum:]_\\- ]", "_", name)
      safe <- trimws(safe)
      safe <- gsub(" ", "_", safe)
      return(tolower(safe))
    },

    ensure_plot_folder = function(plot_type) {
      base_dir <- file.path("..", "Output files")
      plot_dir <- file.path(base_dir, plot_type)
      dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)
      return(plot_dir)
    },

    export_current_plot = function(p, plot_type, plot_name, dpi = 300) {
      folder <- private$ensure_plot_folder(plot_type)
      filename <- paste0(private$sanitize_filename(plot_name), ".png")
      filepath <- file.path(folder, filename)

      ggsave(filename = filepath, plot = p, dpi = dpi, width = 9, height = 7, bg="white")

      if (self$verbose) {
        cat(sprintf("Saved plot to: %s\n", filepath))
      }
    },

    get_feature_columns = function(feature_cols = NULL) {
      if (is.null(feature_cols)) {
        feature_cols <- c("AVG_SPEED", "DISTANCE", "CRS_ELAPSED_TIME")
      }
      return(intersect(feature_cols, names(self$data)))
    },

    categorize_delay_labels = function(labels) {
      case_when(
        labels <= 0  ~ "On-time / Early",
        labels <= 15 ~ "Minor delay",
        labels <= 60 ~ "Moderate delay",
        TRUE         ~ "Long delay"
      )
    },

    generic_categorical_labels = function(labels) {
      as.character(labels)
    }
  )
)

HypothesisTesting <- R6Class("HypothesisTesting",
  public = list(
    data = NULL,
    verbose = NULL,
    alpha = 0.05,

    initialize = function(data, verbose = TRUE) {
      self$data <- as.data.frame(data)
      self$verbose <- verbose
    },

    test_weekend_vs_weekday = function() {
      if (self$verbose) cat("\n======================================================\n")
      if (self$verbose) cat("   HYPOTHESIS TEST 1: Weekend vs. Weekday Delays\n")
      if (self$verbose) cat("======================================================\n")

      if (!all(c("ARR_DELAY", "IS_WEEKEND") %in% names(self$data))) {
        cat("Error: Missing required columns.\n")
        return(invisible(self))
      }

      # Run Welch's t-test
      test_result <- t.test(ARR_DELAY ~ IS_WEEKEND, data = self$data)
      print(test_result)

      # --- Automated Interpretation ---
      cat("\n--- Conclusion ---\n")
      if (test_result$p.value < self$alpha) {
        cat(sprintf("Result: REJECT the null hypothesis (p = %g < 0.05).\n", test_result$p.value))

        # Extract means to see which is higher
        mean_weekday <- test_result$estimate[1] # Group 0
        mean_weekend <- test_result$estimate[2] # Group 1

        if (mean_weekend > mean_weekday) {
          cat("Conclusion: Weekends have significantly HIGHER arrival delays than weekdays.\n")
        } else {
          cat("Conclusion: Weekends have significantly LOWER arrival delays than weekdays.\n")
        }
      } else {
        cat(sprintf("Result: FAIL TO REJECT the null hypothesis (p = %g >= 0.05).\n", test_result$p.value))
        cat("Conclusion: There is no significant difference in delays between weekends and weekdays.\n")
      }

      invisible(self)
    },

    test_pandemic_impact = function() {
      if (self$verbose) cat("\n======================================================\n")
      if (self$verbose) cat("   HYPOTHESIS TEST 2: Pandemic Impact on Delays\n")
      if (self$verbose) cat("======================================================\n")

      if (!all(c("ARR_DELAY", "FL_YEAR") %in% names(self$data))) {
        cat("Error: Missing required columns.\n")
        return(invisible(self))
      }

      # Filter the data and label the eras
      hyp_data <- self$data %>%
        filter(FL_YEAR %in% c(2019, 2022, 2023)) %>%
        mutate(ERA = ifelse(FL_YEAR == 2019, "Pre-Pandemic", "Post-Pandemic"))

      # Extract the delay vectors
      post_delays <- hyp_data %>% filter(ERA == "Post-Pandemic") %>% pull(ARR_DELAY)
      pre_delays <- hyp_data %>% filter(ERA == "Pre-Pandemic") %>% pull(ARR_DELAY)

      if (length(post_delays) == 0 || length(pre_delays) == 0) {
        cat("Error: Not enough data for the specified years to run the test.\n")
        return(invisible(self))
      }

      # Run Welch's one-sided t-test
      test_result <- t.test(post_delays, pre_delays, alternative = "greater")
      print(test_result)

      # --- Automated Interpretation ---
      if (test_result$p.value < self$alpha) {
        cat(sprintf("Result: REJECT the null hypothesis (p = %g < 0.05).\n", test_result$p.value))
        cat("Conclusion: Post-pandemic operations (2022-2023) suffered significantly more delays than pre-pandemic (2019) operations.\n")
      } else {
        cat(sprintf("Result: FAIL TO REJECT the null hypothesis (p = %g >= 0.05).\n", test_result$p.value))
        cat("Conclusion: Post-pandemic delays are NOT significantly higher than pre-pandemic delays.\n")
      }

      invisible(self)
    },

    run_all_tests = function() {
      self$test_weekend_vs_weekday()$
        test_pandemic_impact()

      invisible(self)
    }
  )
)

# ---------------------------------------------------------
# Main Script
# ---------------------------------------------------------

# 1. Load the dataset
cat("\n--- Loading Dataset ---\n")
df_flights <- read.csv("Project Datasets/flights_sample_3m.csv")

# 2. Preprocess the data
cat("\n--- Starting Data Preprocessing ---\n")
processor <- DataPreprocess$new(df_flights, verbose = TRUE)

df_flights_clean <- processor$
  drop_columns()$
  report_missing_values()$
  filter_cancelled_diverted()$
  clean_na()$
  add_date_features()$
  convert_to_season()$
  is_weekend()$
  route()$
  avg_speed()$
  dep_hour()$
  arr_hour()$
  peak_morning()$
  peak_evening()$
  convert_scheduled_times_cyclical()$
  origin_state()$
  dest_state()$
  fix_negative_delays()$
  export_to_csv('Project Datasets/cleaned_flights.csv')$
  get_data()
cat("\n--- Finished Data Preprocessing ---\n")

# 3. Split data and map encodings
cat("\n--- Starting Data Splitting & Encoding ---\n")
splitter <- DataSplit$new(df_flights_clean)
splitter$export_encoding_mappings('Project Datasets/encoding_mappings.csv')
cat("\n--- Finished Data Splitting & Encoding ---\n")

# 4. Run Exploratory Data Analysis
cat("\n--- Running Exploratory Data Analysis ---\n")
eda_instance <- EDA$new(df_flights_clean)
eda_instance$plot_all_core()
cat("\n--- Finished Exploratory Data Analysis ---\n")

# 5. PCA and UMAP
cat("\n--- Running PCA and UMAP ---\n")

X_train <- splitter$data_train
y_train <- splitter$labels_train

pca_features <- c(
  "AVG_SPEED", "DISTANCE", "CRS_ELAPSED_TIME",
  "CRS_DEP_TIME_sin", "CRS_DEP_TIME_cos",
  "CRS_ARR_TIME_sin", "CRS_ARR_TIME_cos"
)

umap_features <- c(
  "AVG_SPEED", "DISTANCE", "CRS_ELAPSED_TIME",
  "CRS_DEP_TIME_sin", "CRS_DEP_TIME_cos",
  "CRS_ARR_TIME_sin", "CRS_ARR_TIME_cos",
  "FL_MONTH", "FL_DAY_OF_WEEK", "SEASON",
  "PEAK_MORNING", "PEAK_EVENING"
)

pca_features <- intersect(pca_features, names(X_train))
umap_features <- intersect(umap_features, names(X_train))

label_views <- list(Delay = y_train)

if ("PEAK_MORNING" %in% names(X_train)) label_views[["Peak Morning"]] <- X_train$PEAK_MORNING
if ("PEAK_EVENING" %in% names(X_train)) label_views[["Peak Evening"]] <- X_train$PEAK_EVENING
if ("SEASON" %in% names(X_train))       label_views[["Season"]] <- X_train$SEASON
if ("FL_MONTH" %in% names(X_train))     label_views[["Month"]] <- X_train$FL_MONTH
if ("FL_DAY_OF_WEEK" %in% names(X_train)) label_views[["Day of Week"]] <- X_train$FL_DAY_OF_WEEK

# --- PCA Loop ---
for (label_name in names(label_views)) {
  cat(sprintf("\n===== PCA (%s) =====\n", label_name))
  labels <- label_views[[label_name]]

  dim_red <- DimReduction$new(data = X_train, labels = labels, verbose = FALSE)
  dim_red$run_pca(feature_cols = pca_features)

  if (label_name == "Delay") {
    dim_red$plot_pca(label_mode = "delay_categorical", export = FALSE)
    dim_red$plot_pca(label_mode = "continuous", export = FALSE)
  } else {
    dim_red$plot_pca(label_mode = "generic_categorical", export = FALSE)
  }
}

# --- UMAP Loop ---
for (label_name in names(label_views)) {
  cat(sprintf("\n===== UMAP (%s) =====\n", label_name))
  labels <- label_views[[label_name]]

  dim_red <- DimReduction$new(data = X_train, labels = labels, verbose = FALSE)
  dim_red$run_umap(feature_cols = umap_features, max_samples = 100000)

  if (label_name == "Delay") {
    dim_red$plot_umap(label_mode = "delay_categorical", export = FALSE)
    dim_red$plot_umap(label_mode = "continuous", export = FALSE)
  } else {
    dim_red$plot_umap(label_mode = "generic_categorical", export = FALSE)
  }
}

cat("\n--- Finished PCA and UMAP ---\n")

# 6. Hypotheses Testing
cat("\n--- Running Hypothesis Testing ---\n")
tester <- HypothesisTesting$new(data = df_flights_clean, verbose = TRUE)
tester$run_all_tests()
cat("\n--- Finished Hypothesis Testing ---\n")