# Load required libraries
required_packages <- c("readxl", "dplyr", "lubridate", "tidyr", "caret", "nnet", "ggplot2", "magrittr", "NeuralNetTools")
new_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages)

lapply(required_packages, require, character.only = TRUE)

# Define the file path
file_path <- "C:/Users/Admin/Downloads/Telegram Desktop/DatasetNN.xlsx"

# Check if the file exists
if (!file.exists(file_path)) {
  stop("The specified file does not exist.")
}

# Read the dataset
DatasetNN <- read_excel(file_path)

# Make column names unique
colnames(DatasetNN) <- make.names(colnames(DatasetNN), unique=TRUE)

# Convert Purchase Date to Date format
DatasetNN$Purchase.Date <- as.Date(DatasetNN$Purchase.Date, format="%Y-%m-%d %H:%M:%S")

# Calculate days since purchase
DatasetNN <- DatasetNN %>%
  mutate(days_since_purchase = as.numeric(Sys.Date() - Purchase.Date))

# Convert Product Category to factor
DatasetNN <- DatasetNN %>%
  mutate(Product.Category = as.factor(Product.Category))

# Aggregate data to find the most purchased product category per customer
most_purchased <- DatasetNN %>%
  group_by(Customer.ID) %>%
  summarise(most_purchased_category = names(which.max(table(Product.Category))))

# Merge the most purchased category with the original dataset
DatasetNN <- left_join(DatasetNN, most_purchased, by = "Customer.ID")

# Check the class distribution before partitioning
print("Class distribution before partitioning:")
print(table(DatasetNN$Product.Category))

# Check the most purchased product category distribution
print("Most purchased product category distribution:")
print(table(DatasetNN$most_purchased_category))

# Calculate the total number of customers who purchased the most
total_customers_most_purchased <- n_distinct(DatasetNN$Customer.ID[DatasetNN$Product.Category == unique(DatasetNN$most_purchased_category)])
cat("Total number of customers who purchased the most:", total_customers_most_purchased, "\n")

# Calculate the total number of customers who purchased the most by each gender
most_purchased_gender <- DatasetNN %>%
  group_by(Gender, most_purchased_category) %>%
  summarise(total_customers = n_distinct(Customer.ID))

# Print the total number of customers who purchased the most by each gender
print("Total number of customers who purchased the most by each gender:")
print(most_purchased_gender)

# Identify the gender with the highest purchase
gender_highest_purchase <- most_purchased_gender %>%
  group_by(Gender) %>%
  summarise(total_customers = sum(total_customers)) %>%
  filter(total_customers == max(total_customers))

# Print the gender with the highest purchase
cat("Gender with the highest purchase:", gender_highest_purchase$Gender, "\n")

# Adjust partitioning to ensure each class has enough records
set.seed(123)
trainIndex <- createDataPartition(DatasetNN$Product.Category, p = .8, list = FALSE, times = 1)
trainData <- DatasetNN[trainIndex,]
testData <- DatasetNN[-trainIndex,]

# Check the class distribution in the training set
print("Class distribution in the training set:")
print(table(trainData$Product.Category))

# If there are any classes with only one level, oversample to balance the training set
one_level_classes <- names(which(table(trainData$Product.Category) < 2))

if (length(one_level_classes) > 0) {
  for (class in one_level_classes) {
    # Identify rows corresponding to the minority class
    minority_rows <- trainData$Product.Category == class
    
    # Count the number of minority samples
    minority_count <- sum(minority_rows)
    
    # Identify rows corresponding to the majority class
    majority_rows <- trainData$Product.Category != class
    
    # Sample from the majority class to match the minority class
    sampled_indices <- sample(which(majority_rows), minority_count, replace = TRUE)
    
    # Append the sampled rows to the training data
    trainData <- rbind(trainData, trainData[sampled_indices, ])
  }
}
# Check the class distribution in the training set after oversampling
print("Class distribution in the training set after oversampling:")
print(table(trainData$Product.Category))

# Load required libraries for plotting
library(ggplot2)
library(dplyr)

# Plot the distribution of product categories and purchases by gender
combined_plot <- ggplot(DatasetNN[!is.na(DatasetNN$most_purchased_category), ], aes(x = most_purchased_category)) +
  geom_bar(aes(fill = Gender), position = "dodge") +
  labs(title = "Total Purchases by Most Popular Category and Gender", x = "Most Popular Category", y = "Total Purchases") +
  scale_fill_manual(values = c("pink", "skyblue")) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

print(combined_plot)

# Prepare data for neural network
nn_data <- DatasetNN %>%
  select(-Customer.ID, -Purchase.Date, -most_purchased_category)

# Preserve the target variable before one-hot encoding
product_category <- nn_data$Product.Category

# One-hot encode categorical variables
nn_data <- dummyVars("~ .", data = nn_data) %>%
  predict(newdata = nn_data) %>%
  as.data.frame()

# Scale numeric features
nn_data <- nn_data %>%
  mutate(across(where(is.numeric), scale))

# Restore the target variable
nn_data$Product.Category <- product_category

# Split the data into training and testing sets
trainIndex <- createDataPartition(nn_data$Product.Category, p = .8, list = FALSE)
trainData <- nn_data[trainIndex,]
testData <- nn_data[-trainIndex,]

# Ensure the factor levels are the same for training and test sets
levels_to_use <- levels(DatasetNN$Product.Category)
trainData$Product.Category <- factor(trainData$Product.Category, levels = levels_to_use)
testData$Product.Category <- factor(testData$Product.Category, levels = levels_to_use)

# Convert the target variable to a factor for the neural network
trainData$Product.Category <- as.factor(trainData$Product.Category)
testData$Product.Category <- as.factor(testData$Product.Category)

# Train a neural network model
set.seed(123)
nn_model <- nnet(Product.Category ~ ., data = trainData, size = 10, decay = 0.01, maxit = 100)

# Print the model summary
print(nn_model)

# Extract and display the weights used in the MLP
weights <- nn_model$wts
print("Weights used in the MLP:")
print(weights)

# Visualize the neural network
library(NeuralNetTools)
plotnet(nn_model)

# Make predictions on the test set
predictions <- predict(nn_model, testData, type = "class")

# Ensure the predictions are factors with the same levels as the actual data
predictions <- factor(predictions, levels = levels_to_use)

# Evaluate the model
confusionMatrix(predictions, testData$Product.Category)

