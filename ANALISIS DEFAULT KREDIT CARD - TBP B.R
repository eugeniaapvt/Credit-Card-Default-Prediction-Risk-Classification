# ================================================================
# Analisis Prediksi Gagal Bayar Kartu Kredit
# Dataset: UCI Credit Card Default (Taiwan, 2005)
# ================================================================


# ── 1. Package ───────────────────────────────────────────────
packages_needed <- c(
  "readxl", "dplyr", "ggplot2", "tidyr", "reshape2",
  "caret", "glmnet",
  "xgboost", "pROC", "doParallel", "parallel"
)

invisible(lapply(packages_needed, function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg)
  library(pkg, character.only = TRUE)
}))

# ── Aktifkan parallel processing ─────────────────────────────
n_cores <- max(1, detectCores() - 1)
cl <- makeCluster(n_cores)
registerDoParallel(cl)
cat("Parallel processing aktif:", n_cores, "core\n")


# ── 2. Load Data ─────────────────────────────────────────────
df <- read_excel("C:/Users/ASUS/Downloads/default+of+credit+card+clients/default of credit card clients.xls", skip = 1)
names(df)[names(df) == "default payment next month"] <- "default"
cat("Dimensi data:", nrow(df), "baris x", ncol(df), "kolom\n")


# ── 3. Eksplorasi Awal ───────────────────────────────────────
cat("\n--- Struktur Data ---\n"); str(df)
cat("\nJumlah missing value:", sum(is.na(df)), "\n")
cat("Jumlah baris duplikat:", sum(duplicated(df)), "\n")


# ── 4. EDA ───────────────────────────────────────────────────

# 4a. Distribusi kelas target
df %>%
  mutate(Status = ifelse(default == 1, "Default (1)", "Tidak Default (0)")) %>%
  ggplot(aes(x = Status, fill = Status)) +
  geom_bar(width = 0.5, show.legend = FALSE) +
  geom_text(stat = "count", aes(label = after_stat(count)), vjust = -0.5, size = 4.5) +
  scale_fill_manual(values = c("Default (1)" = "#E05C5C", "Tidak Default (0)" = "#5C8AE0")) +
  labs(title = "Distribusi Kelas Target", x = "Status Pembayaran", y = "Jumlah Nasabah") +
  theme_minimal(base_size = 13)

# 4b. Heatmap korelasi
melt(cor(df %>% select(where(is.numeric)))) %>%
  ggplot(aes(Var1, Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "#3B6FD4", mid = "white", high = "#D44B3B",
                       midpoint = 0, limits = c(-1, 1)) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1, size = 6.5),
        axis.text.y = element_text(size = 6.5)) +
  labs(title = "Heatmap Korelasi Antar Fitur", x = "", y = "", fill = "r")


# ── 5. Feature Engineering ─────────────────────────────
bersihkan_fitur <- function(data) {
  data %>%
    select(-ID) %>%
    mutate(
      Rata_Tagihan    = rowMeans(select(., BILL_AMT1:BILL_AMT6)),
      Rata_Pembayaran = rowMeans(select(., PAY_AMT1:PAY_AMT6)),
      
      Utilisasi_Max = Rata_Tagihan / (LIMIT_BAL + 1),
      Rasio_Bayar_Tagihan = ifelse(rowSums(select(., BILL_AMT1:BILL_AMT6)) > 0,
                                   rowSums(select(., PAY_AMT1:PAY_AMT6)) /
                                     rowSums(select(., BILL_AMT1:BILL_AMT6)), 0),
      
      Payment_Ratio_Last = PAY_AMT1 / (BILL_AMT1 + 1),
      
      # delay (reduced)
      Mean_Delay    = rowMeans(select(., PAY_0:PAY_6)),
      
      # penting
      Log_Limit = log1p(LIMIT_BAL),
      
      EDUCATION = factor(ifelse(EDUCATION %in% c(0,5,6), 4L, EDUCATION)),
      MARRIAGE  = factor(ifelse(MARRIAGE  %in% c(0,3),   2L, MARRIAGE))
    ) %>%
    select(
      default,
      PAY_0, Mean_Delay,
      Rata_Tagihan, Rata_Pembayaran,
      Utilisasi_Max, Rasio_Bayar_Tagihan, Payment_Ratio_Last,
      Log_Limit,
      AGE, SEX, EDUCATION, MARRIAGE
    )
}
df_clean <- bersihkan_fitur(df)

# ── Cleaning NA / INF ─────────────────────────────
df_clean <- df_clean %>%
  mutate(across(everything(), ~ifelse(is.infinite(.), NA, .))) %>%
  mutate(across(where(is.numeric), ~replace_na(., 0)))

# ── Split fitur & target ───────────────────────────
y_all <- factor(df_clean$default, levels = c(0,1), labels = c("TidakDefault","Default"))
X_all <- df_clean %>% select(-default)

# ── Encoding ───────────────────────────────────────
X_encoded <- model.matrix(~ . - 1, data = X_all) %>% as.data.frame()

cat("\nJumlah fitur setelah FE:", ncol(X_encoded), "\n")

# ── Cek Korelasi ───────────────────────────────────────
cor_data <- df_clean %>% select(where(is.numeric))
cor_mat  <- cor(cor_data)
round(cor_mat, 2)


# ── 6. Train-Test Split (70:30) ──────────────────────────────
set.seed(42)
idx_train   <- createDataPartition(y_all, p = 0.70, list = FALSE)
X_train_raw <- X_encoded[idx_train, ]; X_test_raw <- X_encoded[-idx_train, ]
y_train     <- y_all[ idx_train];       y_test     <- y_all[-idx_train]


# ── 7. Deteksi & Capping Outlier ──────────────────────────────
cols_cek <- c("Rata_Tagihan", "Rata_Pembayaran", "Utilisasi_Max",
              "Payment_Ratio_Last", "Log_Limit", "Rasio_Bayar_Tagihan")

df_outlier <- df_clean %>% select(all_of(cols_cek)) %>% select(where(is.numeric))

# Boxplot outlier
df_outlier %>%
  pivot_longer(everything(), names_to = "Fitur", values_to = "Nilai") %>%
  ggplot(aes(x = Fitur, y = Nilai)) +
  geom_boxplot(fill = "#70C8C8", outlier.color = "#D44B3B",
               outlier.alpha = 0.3, outlier.size = 0.8) +
  facet_wrap(~ Fitur, scales = "free", ncol = 3) +
  labs(title = "Deteksi Outlier per Fitur", x = "", y = "") +
  theme_minimal(base_size = 11) +
  theme(axis.text.x = element_blank())

# Ringkasan IQR
deteksi_outlier <- function(x, nama) {
  q1  <- quantile(x, 0.25, na.rm = TRUE)
  q3  <- quantile(x, 0.75, na.rm = TRUE)
  iqr <- q3 - q1
  data.frame(Fitur       = nama,
             N_Outlier   = sum(x < (q1-1.5*iqr) | x > (q3+1.5*iqr), na.rm=TRUE),
             Pct_Outlier = round(sum(x < (q1-1.5*iqr) | x > (q3+1.5*iqr), na.rm=TRUE) /
                                   length(x) * 100, 2))
}
ringkasan_outlier <- bind_rows(Map(deteksi_outlier,
                                   as.list(df_outlier),
                                   as.list(names(df_outlier)))) %>%
  arrange(desc(Pct_Outlier))
cat("\n===== Ringkasan Outlier =====\n"); print(ringkasan_outlier)

# Capping q99
vars_cap <- intersect(c("Payment_Ratio_Last", "Rata_Pembayaran",
                        "Rata_Tagihan", "Rasio_Bayar_Tagihan"),
                      names(X_train_raw))
for (v in vars_cap) {
  q99 <- quantile(X_train_raw[[v]], 0.99, na.rm = TRUE)
  X_train_raw[[v]] <- pmin(X_train_raw[[v]], q99)
  X_test_raw[[v]]  <- pmin(X_test_raw[[v]],  q99)
}


# ── 8. Normalisasi  ──────────────────────────────
cols_scale <- c("Rata_Tagihan", "Rata_Pembayaran", "Utilisasi_Max",
                "Rasio_Bayar_Tagihan", "Payment_Ratio_Last", "Log_Limit",
                "Mean_Delay")

preproc <- preProcess(X_train_raw[, cols_scale], method = c("center","scale"))

X_train_sc <- X_train_raw
X_test_sc  <- X_test_raw

X_train_sc[, cols_scale] <- predict(preproc, X_train_raw[, cols_scale])
X_test_sc[, cols_scale]  <- predict(preproc, X_test_raw[, cols_scale])

any(is.na(df_clean))
cat("Dimensi X_train_sc:", nrow(X_train_sc), "x", ncol(X_train_sc), "\n")

colnames(X_train_sc)


# ── 9. CROSS VALIDATION ───────────────────────────────
tc_cv <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 3,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  sampling = "up",
  allowParallel = TRUE
)


# ── 10. HYPERPARAMETER TUNING ───────────────────────────────

mk_pred <- function(p, th = 0.5) {
  factor(ifelse(p >= th, "Default", "TidakDefault"),
         levels = c("TidakDefault", "Default"))
}

# ── 11a. Logistic Regression ──────────────────────────────────
cat("\n[1/3] Tuning Logistic Regression...\n")

# Default
model_lr_default <- train(x = X_train_sc, y = y_train,
                          method = "glmnet", trControl = tc_cv, metric = "ROC")

# Tuned
grid_lr <- expand.grid(alpha = c(0, 0.5, 1), lambda = c(0.001, 0.01))
set.seed(42)
model_lr <- train(x = X_train_sc, y = y_train, method = "glmnet",
                  trControl = tc_cv, tuneGrid = grid_lr, metric = "ROC")
cat("  Best → alpha:", model_lr$bestTune$alpha,
    "| lambda:", model_lr$bestTune$lambda, "\n")

# ── 11b. Decision Tree ────────────────────────────────────────
cat("[2/3] Tuning Decision Tree...\n")

# Default
model_dt_default <- train(x = X_train_sc, y = y_train,
                          method = "rpart", trControl = tc_cv, metric = "ROC")

# Tuned
grid_dt <- expand.grid(cp = c(0.0001, 0.001, 0.005, 0.01))
set.seed(42)
model_dt <- train(x = X_train_sc, y = y_train, method = "rpart",
                  trControl = tc_cv, tuneGrid = grid_dt, metric = "ROC",
                  control = rpart::rpart.control(maxdepth = 6, minsplit = 20))
cat("  Best cp:", model_dt$bestTune$cp, "\n")

# ── 11c. XGBoost ──────────────────────────────────────────────
cat("[3/3] Tuning XGBoost...\n")

label_tr <- as.numeric(y_train == "Default")

df_test        <- X_test_sc
df_test$target <- y_test
df_test        <- df_test[complete.cases(df_test), ]
y_test         <- df_test$target
X_test_sc      <- df_test %>% select(-target)
label_te       <- as.numeric(y_test == "Default")

mat_tr <- xgb.DMatrix(data = data.matrix(X_train_sc), label = label_tr)
mat_te <- xgb.DMatrix(data = data.matrix(X_test_sc),  label = label_te)

# Default XGBoost
params_default <- list(objective = "binary:logistic", eval_metric = "auc",
                       max_depth = 6, eta = 0.3, subsample = 1.0,
                       colsample_bytree = 1.0, min_child_weight = 1)
model_xgb_default <- xgb.train(params = params_default, data = mat_tr,
                               nrounds = 100, verbose = 0)

# Tuned XGBoost
xgb_grid <- expand.grid(max_depth = c(6, 9),
                        eta       = c(0.05, 0.1),
                        subsample = c(0.8, 1.0))

spw <- sum(y_train == "TidakDefault") / sum(y_train == "Default")
best_auc <- 0; best_params <- NULL; best_nround <- 100

for (i in seq_len(nrow(xgb_grid))) {
  params_i <- list(
    objective        = "binary:logistic",
    eval_metric      = "auc",
    max_depth        = xgb_grid$max_depth[i],
    eta              = xgb_grid$eta[i],
    subsample        = xgb_grid$subsample[i],
    colsample_bytree = 0.8,
    min_child_weight = 5,
    lambda           = 1,
    alpha            = 0.5,
    scale_pos_weight = spw
  )
  cv_res <- xgb.cv(params = params_i, data = mat_tr,
                   nrounds = 300, nfold = 3,
                   early_stopping_rounds = 30, verbose = 0)
  log      <- cv_res$evaluation_log
  auc_i    <- max(log$test_auc_mean, na.rm = TRUE)
  nround_i <- if (!is.null(cv_res$best_iteration) && cv_res$best_iteration > 0)
    cv_res$best_iteration else which.max(log$test_auc_mean)
  nround_i <- max(nround_i, 30)
  
  if (auc_i > best_auc) {
    best_auc    <- auc_i
    best_params <- params_i
    best_nround <- nround_i
    cat(sprintf("  [%d/%d] Baru: AUC=%.4f | depth=%d eta=%.2f sub=%.1f nround=%d\n",
                i, nrow(xgb_grid), best_auc,
                params_i$max_depth, params_i$eta, params_i$subsample, best_nround))
  }
}

if (is.null(best_params)) {
  cat("  PERINGATAN: CV gagal, pakai parameter default.\n")
  best_params <- list(objective="binary:logistic", eval_metric="auc",
                      max_depth=6, eta=0.05, subsample=0.8,
                      colsample_bytree=0.8, min_child_weight=5,
                      scale_pos_weight=spw)
  best_nround <- 100
}

cat("CV AUC XGBoost terbaik:", round(best_auc, 4), "| nround:", best_nround, "\n")
model_xgb <- xgb.train(params = best_params, data = mat_tr,
                       nrounds = best_nround, verbose = 0)


# ── 12. EVALUASI SEBELUM VS SESUDAH HYPERPARAMETER TUNING ────

# F1 macro
hitung_f1_macro <- function(y_true, y_pred) {
  classes <- levels(y_true)
  f1_per_kelas <- sapply(classes, function(kls) {
    tp <- sum(y_pred == kls & y_true == kls)
    fp <- sum(y_pred == kls & y_true != kls)
    fn <- sum(y_pred != kls & y_true == kls)
    prec   <- ifelse((tp + fp) == 0, 0, tp / (tp + fp))
    recall <- ifelse((tp + fn) == 0, 0, tp / (tp + fn))
    ifelse((prec + recall) == 0, 0, 2 * prec * recall / (prec + recall))
  })
  mean(f1_per_kelas)
}

hitung_metrik <- function(nama, y_true, y_pred, y_prob) {
  
  cm      <- confusionMatrix(y_pred, y_true, positive = "Default")
  auc_val <- as.numeric(auc(roc(as.integer(y_true) - 1L, y_prob, quiet = TRUE)))
  
  recall  <- cm$byClass[["Recall"]]
  spec    <- cm$byClass[["Specificity"]]
  
  # Balanced Accuracy
  bal_acc <- (recall + spec) / 2
  
  # F1 Macro
  f1_macro <- hitung_f1_macro(y_true, y_pred)
  
  data.frame(
    Model            = nama,
    Recall           = round(recall,   4),   # minimasi FN
    BalancedAccuracy = round(bal_acc,  4),   # adil antar kelas
    F1_Macro         = round(f1_macro, 4),   # keseimbangan dua kelas
    ROC_AUC          = round(auc_val,  4)    # performa global
  )
}

eval_dataset <- function(nama, model, X_train, y_train, X_test, y_test, is_xgb = FALSE) {
  if (!is_xgb) {
    prob_tr <- predict(model, X_train, type = "prob")[["Default"]]
    prob_te <- predict(model, X_test,  type = "prob")[["Default"]]
  } else {
    prob_tr <- predict(model, xgb.DMatrix(data.matrix(X_train)))
    prob_te <- predict(model, xgb.DMatrix(data.matrix(X_test)))
  }
  
  pred_tr <- mk_pred(prob_tr)
  pred_te <- mk_pred(prob_te)
  
  tr <- hitung_metrik(nama, y_train, pred_tr, prob_tr); tr$Dataset <- "Train"
  te <- hitung_metrik(nama, y_test,  pred_te, prob_te); te$Dataset <- "Test"
  rbind(tr, te)
}

# Sebelum tuning
hasil_default <- bind_rows(
  eval_dataset("Logistic Regression", model_lr_default,  X_train_sc, y_train, X_test_sc, y_test),
  eval_dataset("Decision Tree",       model_dt_default,  X_train_sc, y_train, X_test_sc, y_test),
  eval_dataset("XGBoost",             model_xgb_default, X_train_sc, y_train, X_test_sc, y_test, is_xgb = TRUE)
)
hasil_default$Tahap <- "Sebelum Hyperparameter Tuning"

# Setelah tuning
hasil_tuned <- bind_rows(
  eval_dataset("Logistic Regression", model_lr,  X_train_sc, y_train, X_test_sc, y_test),
  eval_dataset("Decision Tree",       model_dt,  X_train_sc, y_train, X_test_sc, y_test),
  eval_dataset("XGBoost",             model_xgb, X_train_sc, y_train, X_test_sc, y_test, is_xgb = TRUE)
)
hasil_tuned$Tahap <- "Setelah Hyperparameter Tuning"

hasil_hp <- bind_rows(hasil_default, hasil_tuned) %>%
  arrange(Model, Tahap, Dataset) %>%
  select(Model, Tahap, Dataset, Recall, BalancedAccuracy, F1_Macro, ROC_AUC)

rownames(hasil_hp) <- NULL
cat("\n========== Perbandingan Train vs Test — Sebelum & Sesudah Hyperparameter Tuning ==========\n")
print(hasil_hp)

# Logistic Regression
prob_lr_tuned  <- predict(model_lr, X_test_sc, type = "prob")[["Default"]]
pred_lr_tuned  <- mk_pred(prob_lr_tuned)

# Decision Tree
prob_dt_tuned  <- predict(model_dt, X_test_sc, type = "prob")[["Default"]]
pred_dt_tuned  <- mk_pred(prob_dt_tuned)

# XGBoost
prob_xgb_tuned <- predict(model_xgb, xgb.DMatrix(data.matrix(X_test_sc)))
pred_xgb_tuned <- mk_pred(prob_xgb_tuned)

# ── 12. VISUALISASI EVALUASI ──────────────────────────────────

# 12a. Bar chart 3 metrik utama
hasil_tuned %>%
  select(Model, Recall, BalancedAccuracy, F1_Macro, ROC_AUC) %>%
  pivot_longer(-Model, names_to = "Metrik", values_to = "Nilai") %>%
  mutate(Metrik = recode(Metrik,
                         BalancedAccuracy = "Balanced Accuracy",
                         F1_Macro         = "F1 Macro",
                         ROC_AUC          = "ROC AUC")) %>%
  ggplot(aes(x = Model, y = Nilai, fill = Metrik)) +
  geom_col(position = position_dodge(0.75), width = 0.65) +
  geom_text(aes(label = round(Nilai, 3)),
            position = position_dodge(0.75), vjust = -0.4, size = 3) +
  scale_y_continuous(limits = c(0, 1.08)) +
  scale_fill_brewer(palette = "Set1") +
  labs(title = "Perbandingan Performa Model",
       subtitle = "Metrik: Recall | Balanced Accuracy | F1 Macro | ROC AUC",
       x = "", y = "Nilai Metrik", fill = "Metrik") +
  theme_minimal(base_size = 12) +
  theme(axis.text.x = element_text(angle = 15, hjust = 1))

# 12b. ROC Curve

prob_lr1  <- predict(model_lr,  X_test_sc, type = "prob")[["Default"]]
prob_dt1  <- predict(model_dt,  X_test_sc, type = "prob")[["Default"]]
prob_xgb1 <- predict(model_xgb, mat_te)

roc_lr  <- roc(as.integer(y_test)-1L, prob_lr1,  quiet = TRUE)
roc_dt  <- roc(as.integer(y_test)-1L, prob_dt1,  quiet = TRUE)
roc_xgb <- roc(as.integer(y_test)-1L, prob_xgb1, quiet = TRUE)

plot(roc_lr,  col="#3B6FD4", lwd=2, main="ROC Curve — Perbandingan Model")
plot(roc_dt,  col="#E0943B", lwd=2, add=TRUE)
plot(roc_xgb, col="#C03BD4", lwd=2, add=TRUE)
abline(a=0, b=1, lty=2, col="gray60")
legend("bottomright", bty="n", lwd=2, cex=0.85,
       col    = c("#3B6FD4","#E0943B","#C03BD4"),
       legend = c(paste0("LR  (AUC=", round(auc(roc_lr),3),  ")"),
                  paste0("DT  (AUC=", round(auc(roc_dt),3),  ")"),
                  paste0("XGB (AUC=", round(auc(roc_xgb),3), ")")))

# 12c. Confusion matrix
par(mfrow=c(2,2), mar=c(3,3,3,1))
Map(function(p, nm) fourfoldplot(table(Prediksi=p, Aktual=y_test),
                                 color=c("#E05C5C","#5CBE7A"), main=nm, conf.level=0),
    list(pred_lr_tuned, pred_dt_tuned, pred_xgb_tuned),
    c("Logistic Regression","Decision Tree","XGBoost"))
par(mfrow=c(1,1))

# 12d. Feature Importance XGBoost
xgb.importance(model=model_xgb) %>%
  as.data.frame() %>% slice_head(n=15) %>%
  ggplot(aes(x=reorder(Feature, Gain), y=Gain)) +
  geom_col(fill="#C03BD4", alpha=0.85) + coord_flip() +
  labs(title="15 Fitur Terpenting — XGBoost (Gain)", x="Fitur", y="Gain") +
  theme_minimal(base_size=12)


# ── 13. ESTIMASI PROBABILITAS DEFAULT ───────────────────────

prob_lr  <- predict(model_lr,  X_test_sc, type = "prob")[["Default"]]
prob_dt  <- predict(model_dt,  X_test_sc, type = "prob")[["Default"]]
prob_xgb <- predict(model_xgb, mat_te)

prob_ensemble <- (prob_lr + prob_dt + prob_xgb) / 3

risk_table <- data.frame(
  ID_Test       = seq_len(length(y_test)),
  Aktual        = y_test,
  Prob_LR       = round(prob_lr,  4),
  Prob_DT       = round(prob_dt,  4),
  Prob_XGB      = round(prob_xgb, 4),
  Prob_Ensemble = round(prob_ensemble, 4),
  
  Segmen_Risiko = cut(prob_xgb,
                      breaks = c(0, 0.30, 0.60, 1.0),
                      labels = c("Risiko Rendah", "Risiko Sedang", "Risiko Tinggi"),
                      include.lowest = TRUE)
)

cat("\n========== Estimasi Probabilitas Default (10 baris pertama) ==========\n")
print(head(risk_table, 10))

# ── Ringkasan segmen risiko ───────────────────────────────────
ringkasan_risiko <- risk_table %>%
  group_by(Segmen_Risiko) %>%
  summarise(
    N_Nasabah     = n(),
    N_Aktual_Default = sum(Aktual == "Default"),
    Pct_Default   = round(mean(Aktual == "Default") * 100, 2),
    Rata_Prob_XGB = round(mean(Prob_XGB), 4),
    .groups = "drop"
  )

cat("\n========== Ringkasan Segmen Risiko ==========\n")
print(ringkasan_risiko)

# ── Visualisasi 1: Distribusi probabilitas default per model ─
risk_table %>%
  select(Aktual, Prob_LR, Prob_DT, Prob_XGB) %>%
  pivot_longer(-Aktual, names_to = "Model", values_to = "Probabilitas") %>%
  mutate(Model = recode(Model,
                        Prob_LR  = "Logistic Regression",
                        Prob_DT  = "Decision Tree",
                        Prob_XGB = "XGBoost")) %>%
  ggplot(aes(x = Probabilitas, fill = Aktual)) +
  geom_histogram(bins = 50, alpha = 0.7, position = "identity") +
  facet_wrap(~ Model, ncol = 2) +
  scale_fill_manual(values = c("Default" = "#E05C5C", "TidakDefault" = "#5C8AE0")) +
  theme_minimal()

# ── Visualisasi 2: Distribusi probabilitas per segmen risiko ─
risk_table %>%
  count(Segmen_Risiko, Aktual) %>%
  group_by(Segmen_Risiko) %>%
  mutate(prop = n / sum(n)) %>%
  ggplot(aes(x = Segmen_Risiko, y = prop, fill = Aktual)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = scales::percent(prop, accuracy = 0.1)),
            position = position_stack(vjust = 0.5),
            size = 3.5) +
  scale_fill_manual(values = c("Default" = "#E05C5C",
                               "TidakDefault" = "#5C8AE0")) +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(title = "Proporsi Default Aktual per Segmen Risiko (XGBoost)",
       subtitle = "Semakin tinggi segmen, semakin tinggi proporsi default",
       x = "Segmen Risiko", y = "Proporsi", fill = "Status Aktual") +
  theme_minimal(base_size = 12)

# ── Visualisasi 3: Density plot probabilitas XGBoost ─────────
ggplot(risk_table, aes(x = Prob_XGB, fill = Aktual)) +
  geom_density(alpha = 0.6) +
  scale_fill_manual(values = c("Default" = "#E05C5C", "TidakDefault" = "#5C8AE0")) +
  labs(title = "Density Probabilitas Default — XGBoost",
       x = "P(Default)", y = "Densitas") +
  theme_minimal()

# ── Matikan cluster parallel ──────────────────────────────────
stopCluster(cl)
cat("\nSelesai!\n")
