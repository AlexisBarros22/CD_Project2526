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
required_packages <- c("methods", "ggplot2", "pracma", "patchwork", "gridExtra", "car", "coin", "stats", "utils", "datasets", "dplyr")  # Add more if needed

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

        # Extract Month and Day of Week (1-7, Monday-Sunday)
        self$data <- self$data %>%
          mutate(
            FL_MONTH = as.numeric(format(FL_DATE, "%m")),
            FL_DAY_OF_WEEK = as.numeric(format(FL_DATE, "%u"))
          )

        if (self$verbose) {
          cat("\nDate features extracted:\n")
          print(head(self$data %>% select(FL_DATE, FL_MONTH, FL_DAY_OF_WEEK)))
        }

        self$data <- self$data %>% select(-FL_DATE)
      }

      invisible(self)
    },

    convert_scheduled_times = function() {
      time_cols <- c('CRS_DEP_TIME', 'CRS_ARR_TIME')

      for (col in time_cols) {
        if (col %in% names(self$data)) {
          self$data[[col]] <- as.numeric(self$data[[col]])

          hours <- self$data[[col]] %/% 100
          minutes <- self$data[[col]] %% 100
          self$data[[col]] <- (hours * 60) + minutes

          if (self$verbose) {
            cat(sprintf("\nConverted %s to minutes since midnight:\n", col))
            print(head(self$data[[col]]))
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

    # Placeholders for our custom encoders and scalers
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
      X <- self$data %>% select(-ARR_DELAY)
      y <- self$data$ARR_DELAY

      # Set seed and create train indices to mimic train_test_split
      set.seed(self$random_state)
      n_rows <- nrow(X)
      train_indices <- sample(seq_len(n_rows), size = floor((1 - self$test_size) * n_rows))

      # Implicit copies are created here by standard R subsetting
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
        # Combine unique values from both columns
        all_states_vec <- c()
        for (col in state_cols) all_states_vec <- c(all_states_vec, as.character(self$data_train[[col]]))
        all_states <- sort(unique(all_states_vec))

        # Create a named vector mapping (0-indexed to match Python)
        self$state_mapping <- setNames(seq_along(all_states) - 1, all_states)

        for (col in state_cols) {
          # Map values. Unmapped/unseen values become NA, which we then replace with -1
          self$data_train[[col]] <- unname(self$state_mapping[as.character(self$data_train[[col]])])
          self$data_train[[col]][is.na(self$data_train[[col]])] <- -1

          self$data_test[[col]] <- unname(self$state_mapping[as.character(self$data_test[[col]])])
          self$data_test[[col]][is.na(self$data_test[[col]])] <- -1
        }
      }

      # ---------- 2) Symmetric encoding for route ----------
      if ('ROUTE' %in% names(self$data_train)) {
        canonical_route <- function(route) {
          parts <- strsplit(as.character(route), "_")[[1]]
          if (length(parts) != 2) return(as.character(route))
          return(paste(sort(parts), collapse = "_"))
        }

        train_routes <- sapply(self$data_train$ROUTE, canonical_route, USE.NAMES = FALSE)
        test_routes  <- sapply(self$data_test$ROUTE, canonical_route, USE.NAMES = FALSE)

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

      # Scale train data and save the center/scale (mean/std)
      scaled_train <- scale(self$data_train[cols_present])
      self$scaler_center <- attr(scaled_train, "scaled:center")
      self$scaler_scale <- attr(scaled_train, "scaled:scale")

      self$data_train[cols_present] <- as.data.frame(scaled_train)

      # Scale test data using the train data's center and scale
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

    plot_target_distribution = function(bins = 80, clip_range = c(-60, 180)) {
      if (!"ARR_DELAY" %in% names(self$data)) {
        cat("ARR_DELAY column not found.\n")
        return(invisible(self))
      }

      # Clip values
      plot_data <- data.frame(
        ARR_DELAY = pmax(clip_range[1], pmin(clip_range[2], self$data$ARR_DELAY))
      )

      p <- ggplot(plot_data, aes(x = ARR_DELAY)) +
        geom_histogram(aes(y = after_stat(density)), bins = bins, fill = "dodgerblue", color = "white", alpha = 0.85) +
        geom_density(color = "#005b96", linewidth = 1) +
        labs(title = "Arrival Delay Distribution (Clipped)", x = "Arrival Delay (minutes)", y = "Count") +
        self$base_theme

      print(p)
      invisible(self)
    },

    plot_numeric_distributions = function(columns = NULL, bins = 40) {
      if (is.null(columns)) {
        columns <- names(self$data)[sapply(self$data, is.numeric)]
      }
      columns <- intersect(columns, names(self$data))

      if (length(columns) == 0) {
        cat("No valid numeric columns found.\n")
        return(invisible(self))
      }

      plots <- list()
      for (col in columns) {
        plot_vals <- self$data[[col]]
        if (col == 'ARR_DELAY') {
          plot_vals <- pmax(-60, pmin(180, plot_vals))
        }

        df <- data.frame(val = plot_vals)
        p <- ggplot(df, aes(x = val)) +
          geom_histogram(aes(y = after_stat(density)), bins = bins, fill = "#66c2a5", color = "white", alpha = 0.9) +
          geom_density(color = "#3288bd", linewidth = 0.8) +
          labs(title = sprintf("Distribution of %s", col), x = col, y = "Count") +
          self$base_theme

        plots[[col]] <- p
      }

      # Combine plots using patchwork
      combined_plot <- wrap_plots(plots, ncol = 2)
      print(combined_plot)
      invisible(self)
    },

    plot_boxplots = function(columns = NULL, clip_dict = NULL) {
      if (is.null(columns)) {
        columns <- c('CRS_DEP_TIME', 'CRS_ARR_TIME', 'CRS_ELAPSED_TIME', 'DISTANCE', 'ARR_DELAY',
                     'SEASON', 'FL_DAY_OF_WEEK', 'FL_MONTH', 'AVG_SPEED', 'PEAK_MORNING', 'PEAK_EVENING')
      }

      if (is.null(clip_dict)) {
        clip_dict <- list(
          'CRS_DEP_TIME' = c(0, 1440), 'CRS_ARR_TIME' = c(0, 1440),
          'CRS_ELAPSED_TIME' = c(0, 400), 'DISTANCE' = c(0, 3000),
          'ARR_DELAY' = c(-60, 300), 'SEASON' = c(1, 4),
          'FL_DAY_OF_WEEK' = c(1, 7), 'FL_MONTH' = c(1, 12),
          'AVG_SPEED' = c(1, 6), 'PEAK_MORNING' = c(0, 1),
          'PEAK_EVENING' = c(0, 1)
        )
      }

      existing_cols <- intersect(columns, names(self$data))
      if (length(existing_cols) == 0) {
        cat("No valid columns found for boxplots.\n")
        return(invisible(self))
      }

      plots <- list()
      for (col in existing_cols) {
        plot_vals <- self$data[[col]]
        if (col %in% names(clip_dict)) {
          limits <- clip_dict[[col]]
          plot_vals <- pmax(limits[1], pmin(limits[2], plot_vals))
        }

        df <- data.frame(val = plot_vals)
        p <- ggplot(df, aes(x = val)) +
          geom_boxplot(fill = "#fdae61", color = "#d53e4f", width = 0.5, outlier.shape = NA) +
          labs(title = sprintf("Boxplot of %s (Clipped)", col), x = col) +
          self$base_theme +
          theme(axis.text.y = element_blank(), axis.ticks.y = element_blank())

        plots[[col]] <- p
      }

      combined_plot <- wrap_plots(plots, ncol = 1)
      print(combined_plot)
      invisible(self)
    },

    plot_correlation_heatmap = function() {
      numeric_data <- self$data[, sapply(self$data, is.numeric), drop = FALSE]

      if (ncol(numeric_data) < 2) {
        cat("Not enough numeric columns available for correlation heatmap.\n")
        return(invisible(self))
      }

      corr_matrix <- cor(numeric_data, use = "pairwise.complete.obs")

      # Convert correlation matrix to a flat format for ggplot2
      corr_df <- as.data.frame(as.table(corr_matrix))

      p <- ggplot(corr_df, aes(x = Var1, y = Var2, fill = Freq)) +
        geom_tile(color = "white") +
        geom_text(aes(label = sprintf("%.2f", Freq)), size = 3) +
        scale_fill_gradient2(low = "#4575b4", mid = "white", high = "#d73027", midpoint = 0, limit = c(-1, 1), name = "Corr") +
        labs(title = "Correlation Heatmap", x = "", y = "") +
        self$base_theme +
        theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1))

      print(p)
      invisible(self)
    },

    plot_delay_by_day_of_week = function() {
      if (!all(c('FL_DAY_OF_WEEK', 'ARR_DELAY') %in% names(self$data))) {
        cat("FL_DAY_OF_WEEK and/or ARR_DELAY not found.\n")
        return(invisible(self))
      }

      delay_by_day <- self$data %>%
        group_by(FL_DAY_OF_WEEK) %>%
        summarize(ARR_DELAY = median(ARR_DELAY, na.rm = TRUE), .groups = 'drop')

      p <- ggplot(delay_by_day, aes(x = as.factor(FL_DAY_OF_WEEK), y = ARR_DELAY, fill = as.factor(FL_DAY_OF_WEEK))) +
        geom_col(show.legend = FALSE) +
        scale_fill_viridis_d() +
        labs(title = "Median Arrival Delay by Day of Week", x = "Day of Week", y = "Median Delay (minutes)") +
        self$base_theme

      print(p)
      invisible(self)
    },

    plot_delay_rate_by_day_of_week = function() {
      if (!all(c('FL_DAY_OF_WEEK', 'ARR_DELAY') %in% names(self$data))) {
        cat("FL_DAY_OF_WEEK and/or ARR_DELAY not found.\n")
        return(invisible(self))
      }

      delay_rate <- self$data %>%
        mutate(DELAYED = ifelse(ARR_DELAY > 0, 1, 0)) %>%
        group_by(FL_DAY_OF_WEEK) %>%
        summarize(DELAYED = mean(DELAYED, na.rm = TRUE), .groups = 'drop')

      p <- ggplot(delay_rate, aes(x = as.factor(FL_DAY_OF_WEEK), y = DELAYED, fill = as.factor(FL_DAY_OF_WEEK))) +
        geom_col(show.legend = FALSE) +
        scale_fill_viridis_d(option = "magma") +
        labs(title = "Proportion of Delayed Flights by Day of Week", x = "Day of Week", y = "Delay Rate") +
        self$base_theme

      print(p)
      invisible(self)
    },

    plot_delay_by_month = function() {
      if (!all(c('FL_MONTH', 'ARR_DELAY') %in% names(self$data))) {
        cat("FL_MONTH and/or ARR_DELAY not found.\n")
        return(invisible(self))
      }

      delay_by_month <- self$data %>%
        group_by(FL_MONTH) %>%
        summarize(ARR_DELAY = median(ARR_DELAY, na.rm = TRUE), .groups = 'drop')

      p <- ggplot(delay_by_month, aes(x = FL_MONTH, y = ARR_DELAY)) +
        geom_line(color = "#008080", linewidth = 1.5) +
        geom_point(color = "darkslategray", size = 3) +
        scale_x_continuous(breaks = 1:12) +
        labs(title = "Median Arrival Delay by Month", x = "Month", y = "Median Delay (minutes)") +
        self$base_theme

      print(p)
      invisible(self)
    },

    plot_delay_vs_distance = function() {
      if (!all(c('DISTANCE', 'ARR_DELAY') %in% names(self$data))) {
        cat("DISTANCE and/or ARR_DELAY not found.\n")
        return(invisible(self))
      }

      df <- data.frame(
        DISTANCE = self$data$DISTANCE,
        ARR_DELAY = pmax(-60, pmin(180, self$data$ARR_DELAY))
      )

      p <- ggplot(df, aes(x = DISTANCE, y = ARR_DELAY)) +
        geom_bin2d(bins = 50) +
        scale_fill_viridis_c(name = "Flights") +
        labs(title = "Arrival Delay vs Distance", x = "Distance (miles)", y = "Arrival Delay (minutes, clipped)") +
        self$base_theme

      print(p)
      invisible(self)
    },

    plot_delay_vs_elapsed_time = function() {
      if (!all(c('CRS_ELAPSED_TIME', 'ARR_DELAY') %in% names(self$data))) {
        cat("CRS_ELAPSED_TIME and/or ARR_DELAY not found.\n")
        return(invisible(self))
      }

      df <- data.frame(
        CRS_ELAPSED_TIME = self$data$CRS_ELAPSED_TIME,
        ARR_DELAY = pmax(-60, pmin(180, self$data$ARR_DELAY))
      )

      p <- ggplot(df, aes(x = CRS_ELAPSED_TIME, y = ARR_DELAY)) +
        geom_bin2d(bins = 45) +
        scale_fill_viridis_c(option = "plasma", name = "Flights") +
        labs(title = "Arrival Delay vs Scheduled Elapsed Time", x = "Scheduled Elapsed Time (minutes)", y = "Arrival Delay (minutes, clipped)") +
        self$base_theme

      print(p)
      invisible(self)
    },

    plot_top_origin_cities_by_average_delay = function(top_n = 15, min_flights = 2000) {
      if (!all(c('ORIGIN_CITY', 'ARR_DELAY') %in% names(self$data))) {
        cat("ORIGIN_CITY and/or ARR_DELAY not found.\n")
        return(invisible(self))
      }

      city_stats <- self$data %>%
        group_by(ORIGIN_CITY) %>%
        summarize(avg_delay = mean(ARR_DELAY, na.rm = TRUE), flights = n(), .groups = 'drop') %>%
        filter(flights >= min_flights) %>%
        arrange(desc(avg_delay)) %>%
        head(top_n)

      p <- ggplot(city_stats, aes(x = avg_delay, y = reorder(ORIGIN_CITY, avg_delay), fill = avg_delay)) +
        geom_col(show.legend = FALSE) +
        scale_fill_viridis_c(option = "magma") +
        labs(title = sprintf("Top %d Origin Cities by Average Delay", top_n), x = "Average Arrival Delay (minutes)", y = "Origin City") +
        self$base_theme

      print(p)
      invisible(self)
    },

    plot_top_dest_cities_by_average_delay = function(top_n = 15, min_flights = 2000) {
      if (!all(c('DEST_CITY', 'ARR_DELAY') %in% names(self$data))) {
        cat("DEST_CITY and/or ARR_DELAY not found.\n")
        return(invisible(self))
      }

      city_stats <- self$data %>%
        group_by(DEST_CITY) %>%
        summarize(avg_delay = mean(ARR_DELAY, na.rm = TRUE), flights = n(), .groups = 'drop') %>%
        filter(flights >= min_flights) %>%
        arrange(desc(avg_delay)) %>%
        head(top_n)

      p <- ggplot(city_stats, aes(x = avg_delay, y = reorder(DEST_CITY, avg_delay), fill = avg_delay)) +
        geom_col(show.legend = FALSE) +
        scale_fill_viridis_c(option = "mako") +
        labs(title = sprintf("Top %d Destination Cities by Average Delay", top_n), x = "Average Arrival Delay (minutes)", y = "Destination City") +
        self$base_theme

      print(p)
      invisible(self)
    },

    plot_top_airlines_by_average_delay = function(top_n = 15, min_flights = 5000) {
      if (!all(c('DOT_CODE', 'ARR_DELAY') %in% names(self$data))) {
        cat("DOT_CODE and/or ARR_DELAY not found.\n")
        return(invisible(self))
      }

      airline_stats <- self$data %>%
        group_by(DOT_CODE) %>%
        summarize(avg_delay = mean(ARR_DELAY, na.rm = TRUE), flights = n(), .groups = 'drop') %>%
        filter(flights >= min_flights) %>%
        arrange(desc(avg_delay)) %>%
        head(top_n)

      p <- ggplot(airline_stats, aes(x = avg_delay, y = reorder(as.character(DOT_CODE), avg_delay), fill = avg_delay)) +
        geom_col(show.legend = FALSE) +
        scale_fill_viridis_c(option = "rocket") +
        labs(title = sprintf("Top %d Airlines (DOT_CODE) by Average Delay", top_n), x = "Average Arrival Delay (minutes)", y = "DOT_CODE") +
        self$base_theme

      print(p)
      invisible(self)
    },

    plot_top_routes_by_average_delay = function(top_n = 15, min_flights = 2000) {
      if (!all(c('ROUTE', 'ARR_DELAY') %in% names(self$data))) {
        cat("ROUTE and/or ARR_DELAY not found.\n")
        return(invisible(self))
      }

      route_stats <- self$data %>%
        group_by(ROUTE) %>%
        summarize(avg_delay = mean(ARR_DELAY, na.rm = TRUE), flights = n(), .groups = 'drop') %>%
        filter(flights >= min_flights) %>%
        arrange(desc(avg_delay)) %>%
        head(top_n)

      p <- ggplot(route_stats, aes(x = avg_delay, y = reorder(ROUTE, avg_delay), fill = avg_delay)) +
        geom_col(show.legend = FALSE) +
        scale_fill_viridis_c() +
        labs(title = sprintf("Top %d Routes by Average Delay", top_n), x = "Average Arrival Delay (minutes)", y = "Route") +
        self$base_theme

      print(p)
      invisible(self)
    },

    plot_top_origin_sta_by_average_delay = function(top_n = 15, min_flights = 2000) {
      if (!all(c('ORIGIN_STATE', 'ARR_DELAY') %in% names(self$data))) {
        cat("ORIGIN_STATE and/or ARR_DELAY not found.\n")
        return(invisible(self))
      }

      state_stats <- self$data %>%
        group_by(ORIGIN_STATE) %>%
        summarize(avg_delay = mean(ARR_DELAY, na.rm = TRUE), flights = n(), .groups = 'drop') %>%
        filter(flights >= min_flights) %>%
        arrange(desc(avg_delay)) %>%
        head(top_n)

      p <- ggplot(state_stats, aes(x = avg_delay, y = reorder(ORIGIN_STATE, avg_delay), fill = avg_delay)) +
        geom_col(show.legend = FALSE) +
        scale_fill_viridis_c(option = "magma") +
        labs(title = sprintf("Top %d Origin States by Average Delay", top_n), x = "Average Arrival Delay (minutes)", y = "Origin State") +
        self$base_theme

      print(p)
      invisible(self)
    },

    plot_top_dest_state_by_average_delay = function(top_n = 15, min_flights = 2000) {
      if (!all(c('DEST_STATE', 'ARR_DELAY') %in% names(self$data))) {
        cat("DEST_STATE and/or ARR_DELAY not found.\n")
        return(invisible(self))
      }

      state_stats <- self$data %>%
        group_by(DEST_STATE) %>%
        summarize(avg_delay = mean(ARR_DELAY, na.rm = TRUE), flights = n(), .groups = 'drop') %>%
        filter(flights >= min_flights) %>%
        arrange(desc(avg_delay)) %>%
        head(top_n)

      p <- ggplot(state_stats, aes(x = avg_delay, y = reorder(DEST_STATE, avg_delay), fill = avg_delay)) +
        geom_col(show.legend = FALSE) +
        scale_fill_viridis_c(option = "mako") +
        labs(title = sprintf("Top %d Destination States by Average Delay", top_n), x = "Average Arrival Delay (minutes)", y = "Destination State") +
        self$base_theme

      print(p)
      invisible(self)
    },

    plot_all_core = function() {
      self$summary()$
        plot_target_distribution()$
        plot_numeric_distributions(columns = c(
          'CRS_DEP_TIME', 'CRS_ARR_TIME', 'CRS_ELAPSED_TIME',
          'DISTANCE', 'SEASON', 'FL_DAY_OF_WEEK', 'FL_MONTH',
          'AVG_SPEED', 'PEAK_MORNING', 'PEAK_EVENING'
        ))$
        plot_boxplots(columns = c(
          'CRS_DEP_TIME', 'CRS_ARR_TIME', 'CRS_ELAPSED_TIME',
          'DISTANCE', 'SEASON', 'FL_DAY_OF_WEEK', 'FL_MONTH',
          'ARR_DELAY', 'AVG_SPEED', 'PEAK_MORNING', 'PEAK_EVENING'
        ))$
        plot_correlation_heatmap()$
        plot_delay_by_day_of_week()$
        plot_delay_rate_by_day_of_week()$
        plot_delay_by_month()$
        plot_delay_vs_distance()$
        plot_delay_vs_elapsed_time()$
        plot_top_origin_cities_by_average_delay()$
        plot_top_dest_cities_by_average_delay()$
        plot_top_airlines_by_average_delay()$
        plot_top_routes_by_average_delay()$
        plot_top_origin_sta_by_average_delay()$
        plot_top_dest_state_by_average_delay()

      invisible(self)
    }
  )
)


# ---------------------------------------------------------
# Main Script
# ---------------------------------------------------------

# 1. Load the dataset
# Alternatively, if you just want to load it directly in base R:
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
  convert_scheduled_times()$
  convert_to_season()$
  route()$
  avg_speed()$
  dep_hour()$
  arr_hour()$
  peak_morning()$
  peak_evening()$
  origin_state()$
  dest_state()$
  export_to_csv('Project Datasets/cleaned_flights.csv')$
  get_data()

# 3. Split data and map encodings
cat("\n--- Starting Data Splitting & Encoding ---\n")
splitter <- DataSplit$new(df_flights_clean)
splitter$export_encoding_mappings('Project Datasets/encoding_mappings.csv')

# 4. Run Exploratory Data Analysis
cat("\n--- Running Exploratory Data Analysis ---\n")
eda_instance <- EDA$new(df_flights_clean)
eda_instance$plot_all_core()