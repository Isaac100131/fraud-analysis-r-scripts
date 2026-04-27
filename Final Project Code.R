#install.packages("caret")
#install.packages("pROC")
library(caret)
library(pROC)

# Load data
df <- read.csv("new_bank_fraud_detection.csv", stringsAsFactors = FALSE)

# Remove identifier columns
df <- df[, !(names(df) %in% c("Transaction_ID", "Merchant_ID", "X"))]

# --------------------------------------------
# Clean variables needed for feature creation
# --------------------------------------------
df$Transaction_Device <- factor(trimws(df$Transaction_Device))
df$Merchant_Category <- factor(trimws(df$Merchant_Category))

# Convert response to factor for classification
df$Is_Fraud <- factor(df$Is_Fraud, levels = c(0, 1), labels = c("NotFraud", "Fraud"))

# Make sure numeric fields are numeric
df$Transaction_Amount <- as.numeric(df$Transaction_Amount)
df$Account_Balance <- as.numeric(df$Account_Balance)

# Remove rows with missing values in required fields
df <- df[!is.na(df$Transaction_Device) &
           !is.na(df$Merchant_Category) &
           !is.na(df$Transaction_Amount) &
           !is.na(df$Account_Balance) &
           !is.na(df$Is_Fraud), ]

# --------------------------------------------
# 1. Merchant-Device Interaction Feature
# Example: "POS_Groceries", "ATM_Electronics"
# --------------------------------------------
df$Merchant_Device_Interact <- interaction(df$Transaction_Device,
                                           df$Merchant_Category,
                                           sep = "_",
                                           drop = TRUE)

df$Merchant_Device_Interact <- factor(df$Merchant_Device_Interact)

# --------------------------------------------
# 2. Amount-to-Balance Ratio Feature
# Captures how large the transaction is
# relative to account balance
# --------------------------------------------
df$Amount_Balance_Ratio <- df$Transaction_Amount / (df$Account_Balance + 1e-6)

# Optional draining-behavior flag
df$High_Drain_Flag <- ifelse(df$Amount_Balance_Ratio > 0.75, 1, 0)
df$High_Drain_Flag <- factor(df$High_Drain_Flag)

# --------------------------------------------
# Remove any factor with fewer than 2 levels
# --------------------------------------------
df <- df[, sapply(df, function(x) !(is.factor(x) && nlevels(x) < 2))]

# --------------------------------------------
# Logistic regression -- Advanced Features
# --------------------------------------------
fraud_glm_adv <- glm(
  Is_Fraud ~ Merchant_Device_Interact + Amount_Balance_Ratio + High_Drain_Flag,
  data = df,
  family = binomial
)

summary(fraud_glm_adv)

# Odds ratios
exp(coef(fraud_glm_adv))

# ----------------------------
# 8. Predicted probabilities
# ----------------------------
prob <- predict(fraud_glm_adv, type = "response")

# Class predictions using 0.05 threshold
pred <- ifelse(prob >= 0.05, "Fraud", "NotFraud")
pred <- factor(pred, levels = c("NotFraud", "Fraud"))

# ----------------------------
# 9. Confusion matrix
# ----------------------------
cm <- confusionMatrix(pred, df$Is_Fraud, positive = "Fraud")
print(cm)

# ----------------------------
# 10. F1 score
# ----------------------------
precision <- cm$byClass["Pos Pred Value"]
recall <- cm$byClass["Sensitivity"]

if (is.na(precision) || is.na(recall) || (precision + recall) == 0) {
  F1 <- 0
} else {
  F1 <- 2 * (precision * recall) / (precision + recall)
}

cat("\nPrecision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1 Score:", F1, "\n")

# ----------------------------
# 11. ROC curve and AUC
# ----------------------------
roc_obj <- roc(response = df$Is_Fraud,
               predictor = prob,
               levels = c("NotFraud", "Fraud"))

auc_value <- auc(roc_obj)

cat("\nAUC:", auc(roc_obj), "\n")

plot(roc_obj,
     main = paste("ROC Curve - Logistic Regression (AUC =", round(auc_value, 3), ")"))

# Best threshold from ROC
coords(roc_obj, "best", ret = "threshold")

# ----------------------------
# 12. 5-fold cross validation
# ----------------------------
ctrl <- trainControl(method = "cv",
                     number = 5,
                     classProbs = TRUE,
                     summaryFunction = twoClassSummary)

cv_model <- train(Is_Fraud ~ Merchant_Device_Interact + Amount_Balance_Ratio + High_Drain_Flag,
                  data = df,
                  method = "glm",
                  family = "binomial",
                  metric = "ROC",
                  trControl = ctrl)

print(cv_model)
print(cv_model$results)
