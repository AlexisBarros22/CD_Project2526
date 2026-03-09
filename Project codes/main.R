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
required_packages <- c("methods", "ggplot2", "pracma", "patchwork", "gridExtra", "car", "coin", "stats", "utils", "datasets", "dplyr", "randomForest", "patchwork")  # Add more if needed

# Check and install missing packages
check_install_packages(required_packages)
library(dplyr)
library(R6)
library(randomForest)
library(ggplot2)
library(gridExtra)
library(pracma)
library(car)
library(coin)
library(stats)

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

    initialize = function(data, verbose = TRUE) {
      # Em R, os data.frames são passados por valor, por isso
      # a atribuição já cria naturalmente uma cópia independente (o equivalente a .copy())
      self$data <- data
      self$verbose <- verbose
    },

    drop_columns = function() {
      columns_to_drop <- c(
        'DEP_DELAY', 'DELAY_DUE_CARRIER', 'DELAY_DUE_WEATHER', 'DELAY_DUE_SECURITY',
        'DELAY_DUE_NAS', 'DELAY_DUE_LATE_AIRCRAFT', 'ARR_TIME', 'DEP_TIME',
        'WHEELS_OFF', 'WHEELS_ON', 'TAXI_OUT', 'TAXI_IN', 'ELAPSED_TIME',
        'AIR_TIME', 'CANCELLATION_CODE', 'AIRLINE', 'AIRLINE_CODE', 'AIRLINE_DOT',
        'FL_NUMBER', 'ORIGIN', 'DEST'
      )

      # any_of() ignora colunas que não existam (o equivalente a errors="ignore")
      self$data <- self$data %>% select(-any_of(columns_to_drop))

      if (self$verbose) {
        print(head(self$data))
      }

      return(invisible(self)) # Permite encadeamento (chaining)
    },

    report_missing_values = function() {
      if (!self$verbose) return(invisible(self))

      cat(sprintf("Total flights: %d\n", nrow(self$data)))
      cat("\nNA values per column:\n")
      print(colSums(is.na(self$data)))

      # Conta as linhas que não estão "completas" (têm pelo menos 1 NA)
      na_rows <- sum(!complete.cases(self$data))
      cat(sprintf("\nTotal rows with at least one NA value: %d\n", na_rows))
      cat(sprintf("Percentage of rows with NA: %.2f%%\n", (na_rows / nrow(self$data) * 100)))

      # Sumários de Cancelados / Divergidos
      if ("CANCELLED" %in% names(self$data)) {
        cat(sprintf("\nCancelled flights: %d\n", sum(self$data$CANCELLED, na.rm = TRUE)))
      }
      if ("DIVERTED" %in% names(self$data)) {
        cat(sprintf("Diverted flights: %d\n", sum(self$data$DIVERTED, na.rm = TRUE)))
      }

      if ("CANCELLED" %in% names(self$data)) {
        cancelled <- self$data %>% filter(CANCELLED == 1)
        if (nrow(cancelled) > 0) {
          cat("\nNA values in CANCELLED flights:\n")
          print(colSums(is.na(cancelled)))
        }
      }

      if ("DIVERTED" %in% names(self$data)) {
        diverted <- self$data %>% filter(DIVERTED == 1)
        if (nrow(diverted) > 0) {
          cat("\nNA values in DIVERTED flights:\n")
          print(colSums(is.na(diverted)))
        }
      }

      return(invisible(self))
    },

    filter_cancelled_diverted = function() {
      if (self$verbose) {
        cat("\n", strrep("=", 60), "\n", sep = "")
        cat("NOW filtering out cancelled/diverted flights...\n")
        cat(strrep("=", 60), "\n")
      }

      # Verifica se as duas colunas existem
      if (all(c("CANCELLED", "DIVERTED") %in% names(self$data))) {
        self$data <- self$data %>% filter(CANCELLED == 0, DIVERTED == 0)

        if (self$verbose) {
          cat(sprintf("\nTotal flights after filtering: %d\n", nrow(self$data)))
        }

        # Remove as colunas CANCELLED e DIVERTED
        self$data <- self$data %>% select(-any_of(c("CANCELLED", "DIVERTED")))
      }

      return(invisible(self))
    },

    clean_na = function() {
      if (self$verbose) {
        cat("\n", strrep("=", 60), "\n", sep = "")
        cat("Number of NA values before dropping:\n")
        print(colSums(is.na(self$data)))
        cat(strrep("=", 60), "\n")
      }

      # O equivalente ao dropna()
      self$data <- na.omit(self$data)

      if (self$verbose) {
        cat(sprintf("\nTotal flights after dropping NA: %d\n", nrow(self$data)))
      }

      return(invisible(self))
    },

    timestamp_to_datetime = function() {
      if ("FL_DATE" %in% names(self$data)) {
        # O equivalente a pd.to_datetime com errors='coerce'
        self$data$FL_DATE <- as.POSIXct(
          self$data$FL_DATE,
          tryFormats = c("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y"),
          optional = TRUE
        )
      }
      return(invisible(self))
    },

    timestamp_to_date = function() {
      if ("FL_DATE" %in% names(self$data)) {

        # 1. Converter a data para formato normal (Date)
        # O tryFormats lida com "2019-01-09"
        self$data$FL_DATE <- as.Date(self$data$FL_DATE, format = "%Y-%m-%d")

        # 2. Extrair ano, mes e dia convertendo temporariamente para POSIXlt
        dates_lt <- as.POSIXlt(self$data$FL_DATE)

        self$data$FL_YEAR <- dates_lt$year + 1900
        self$data$FL_MONTH <- dates_lt$mon + 1
        self$data$FL_DAY <- dates_lt$mday

        # 3. Remover o formato POSIXlt complexo se ainda estiver lá
        # Isto garante que o objeto volta a ser um data.frame limpo.
        self$data <- as.data.frame(self$data)

        if (self$verbose) {
          cat("\nDate components extracted:\n")
          print(head(self$data %>% select(FL_DATE, FL_YEAR, FL_MONTH, FL_DAY)))
        }
      }
      return(invisible(self))
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

    data_train = NULL,
    labels_train = NULL,
    data_test = NULL,
    labels_test = NULL,

    numeric_cols = NULL,
    # Em R não temos um "StandardScaler()" como objeto por defeito,
    # por isso guardamos as métricas usadas para escalar o treino:
    scaler_center = NULL,
    scaler_scale = NULL,

    initialize = function(data, test_size = 0.2, random_state = 48) {
      self$data <- data
      self$test_size <- test_size
      self$random_state <- random_state

      private$load_data()
    }
  ),

  private = list(
    load_data = function() {
      tryCatch({
        # 1. Garantir que os dados são um data.frame puro
        df_safe <- as.data.frame(self$data)

        # 2. Separar X e y
        X <- df_safe %>% select(-ARR_DELAY)
        y <- df_safe$ARR_DELAY

        # --- Train/Test Split ---
        set.seed(self$random_state)

        n_total <- nrow(X)
        n_train <- floor((1 - self$test_size) * n_total)

        train_indices <- sample(seq_len(n_total), size = n_train)

        X_train <- X[train_indices, , drop = FALSE]
        X_test  <- X[-train_indices, , drop = FALSE]
        y_train <- y[train_indices]
        y_test  <- y[-train_indices]

        # --- Identificar Colunas Numéricas ---
        self$numeric_cols <- names(X_train)[sapply(X_train, is.numeric)]

        X_train_scaled <- X_train
        X_test_scaled  <- X_test

        # --- Fit & Transform no Treino ---
        if (length(self$numeric_cols) > 0) {

          # A CORREÇÃO ESTÁ AQUI: Parênteses simples e com vírgula!
          scaled_train <- scale(X_train[, self$numeric_cols, drop = FALSE])

          self$scaler_center <- attr(scaled_train, "scaled:center")
          self$scaler_scale  <- attr(scaled_train, "scaled:scale")

          X_train_scaled[, self$numeric_cols] <- as.data.frame(scaled_train)

          # --- Transform no Teste ---
          scaled_test <- scale(
            X_test[, self$numeric_cols, drop = FALSE],
            center = self$scaler_center,
            scale = self$scaler_scale
          )

          X_test_scaled[, self$numeric_cols] <- as.data.frame(scaled_test)
        }

        # Guardar nas variáveis públicas
        self$data_train <- X_train_scaled
        self$labels_train <- y_train
        self$data_test <- X_test_scaled
        self$labels_test <- y_test

        cat(sprintf(
          "Data split successful: %d training samples, %d testing samples. Scaled %d numeric columns.\n",
          nrow(X_train), nrow(X_test), length(self$numeric_cols)
        ))

      }, error = function(e) {
        cat("Error during data split: ", conditionMessage(e), "\n")
      })
    }
  )
)

EDA <- R6Class("EDA",
  public = list(
    splitter = NULL,

    # Inicialização (equivalente ao __init__)
    initialize = function(splitter) {
      self$splitter <- splitter
    },

    # Equivalente ao perform_eda
    perform_eda = function() {
      cat("Exploratory Data Analysis (EDA) Report:\n")
      cat("--------------------------------------\n")

      cat("\nSummary Statistics for train data:\n")
      print(summary(self$splitter$data_train))

      cat("\nSummary Statistics for test data:\n")
      print(summary(self$splitter$data_test))

      cat("\nDistribution Analysis:\n")
      self$plot_distributions()

      cat("\nCorrelation Analysis:\n")
      self$plot_correlation_heatmap()

      cat("\nFeature Importance Analysis:\n")
      self$plot_feature_importance()
    },

    # Equivalente ao plot_distributions
    plot_distributions = function() {
      train_data <- self$splitter$data_train

      # Em R, em vez de criar uma figura gigante de uma vez,
      # é comum iterar e imprimir cada gráfico sequencialmente
      for (feature in names(train_data)) {
        # Validar se a coluna é numérica antes de fazer o histograma
        if (is.numeric(train_data[[feature]])) {

          # .data[[feature]] é a forma moderna no ggplot2 de passar strings como variáveis
          p <- ggplot(train_data, aes(x = .data[[feature]])) +
            geom_histogram(bins = 30, fill = "steelblue", color = "black") +
            labs(title = paste("Distribution of", feature),
                 x = feature,
                 y = "Frequency") +
            theme_minimal()

          print(p)
        }
      }
    },

    # Equivalente ao plot_correlation_heatmap
    plot_correlation_heatmap = function() {
      # 1. Garantir que é um data.frame clássico
      data_with_labels <- as.data.frame(self$splitter$data_train)

      # 2. Adicionar o target (label)
      data_with_labels$TARGET <- as.numeric(as.character(self$splitter$labels_train))

      # 3. Identificar colunas numéricas usando R Base (à prova de falhas do dplyr em R6)
      is_num <- sapply(data_with_labels, is.numeric)
      num_data <- data_with_labels[, is_num, drop = FALSE]

      if (ncol(num_data) < 2) {
        cat("\n[Aviso] Nao existem colunas numericas suficientes para o Heatmap de Correlacao!\n")
        cat("Verifica se as tuas colunas não foram todas transformadas em 'character' ou 'factor'.\n")
        return(invisible(self))
      }
      # ----------------------------

      # 4. Calcular correlação
      corr_matrix <- cor(as.matrix(num_data), use = "pairwise.complete.obs")

      # Transformar a matriz num formato longo para o ggplot
      corr_df <- as.data.frame(as.table(corr_matrix))
      names(corr_df) <- c("Feature1", "Feature2", "Correlation")

      # Criar o Heatmap
      p <- ggplot(corr_df, aes(x = Feature1, y = Feature2, fill = Correlation)) +
        geom_tile(color = "white") +
        scale_fill_gradient2(low = "blue", high = "red", mid = "white",
                             midpoint = 0, limit = c(-1, 1), space = "Lab",
                             name = "Correlation", na.value = "grey50") +
        theme_minimal() +
        theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1, size = 10),
              axis.text.y = element_text(size = 10)) +
        labs(title = "Correlation Heatmap between Features and Labels",
             x = "", y = "")

      print(p)
    },

    # Equivalente ao plot_feature_importance
    plot_feature_importance = function(n_estimators = 5) {
      y_train <- self$splitter$labels_train
      X_train <- as.data.frame(self$splitter$data_train)

      if(is.list(y_train) || is.data.frame(y_train)) y_train <- y_train[[1]]
      y_train <- as.numeric(as.character(y_train))

      valid_cols <- unlist(lapply(X_train, is.numeric))
      X_train <- X_train[, valid_cols, drop = FALSE]

      valid_idx <- unlist(!is.na(y_train))
      y_train <- y_train[valid_idx]
      X_train <- X_train[valid_idx, , drop = FALSE]

      if (nrow(X_train) == 0 || ncol(X_train) == 0) {
        cat("\n[Aviso] Nao existem dados validos para treinar a Random Forest!\n")
        return(invisible(self))
      }

      max_samples <- 50000 # Limite de linhas para a EDA

      if (nrow(X_train) > max_samples) {
        cat(sprintf("\nReduzindo de %d para %d amostras aleatorias para treinar rapidamente...\n",
                    nrow(X_train), max_samples))
        set.seed(48)
        sample_idx <- sample(seq_len(nrow(X_train)), size = max_samples)

        X_train <- X_train[sample_idx, , drop = FALSE]
        y_train <- y_train[sample_idx]
      }
      # ------------------------------------------

      cat(sprintf("\nA treinar Random Forest (Regressao) com %d amostras e %d features...\n",
                  nrow(X_train), ncol(X_train)))

      rf_model <- randomForest(x = X_train,
                               y = y_train,
                               ntree = n_estimators,
                               importance = TRUE)

      imp_matrix <- importance(rf_model)
      imp_df <- data.frame(
        Feature = rownames(imp_matrix),
        Importance = imp_matrix[, 1]
      )

      imp_df$Feature <- factor(imp_df$Feature, levels = imp_df$Feature[order(imp_df$Importance)])

      p <- ggplot(imp_df, aes(x = Feature, y = Importance)) +
        geom_col(fill = "darkorange") +
        coord_flip() +
        theme_minimal() +
        labs(title = "Feature Importance (Random Forest Regressor)",
             x = "Features",
             y = "Importance (Increase in MSE)")

      print(p)
    }
  )
)


# 1. Execução: Load Data
loader <- DataLoader$new()
df_flights <- loader$data

# 2. Execução: Preprocess Data
processor <- DataPreprocess$new(data = df_flights, verbose = TRUE)
df_flights_clean <- processor$
  drop_columns()$
  report_missing_values()$
  filter_cancelled_diverted()$
  clean_na()$
  timestamp_to_datetime()$
  timestamp_to_date()$
  # encode_cyclical_time()$
  get_data()

# 3. Execução: Data Split
splitter <- DataSplit$new(data = df_flights_clean)

# 4. Execução: EDA
eda_instance <- EDA$new(splitter = splitter)
eda_instance$perform_eda()