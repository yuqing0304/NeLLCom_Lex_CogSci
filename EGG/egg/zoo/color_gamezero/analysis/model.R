library(lme4)
library(lmerTest)
library(readr)
library(ggplot2)
library(ggrepel)

##### DATA PREPROCESSING

# Load and clean the data
data <- read_csv("path/to/your/model_data.csv")  # Path to the CSV output from the processing script
names(data)[names(data) == "informativeness"] <- 'informativeness'
names(data)[names(data) == "hard_diff"] <- 'context.ease'  # Rename for clarity

# Make sure the necessary columns are factors
data$condition = as.factor(data$condition)  # Assuming you have a 'condition' column

# View the dataset
head(data)

##### LINEAR MIXED-EFFECTS MODEL - CONTEXT EASE PREDICTING WORD INFORMATIVENESS

# Mixed model with context ease (hard_diff) as the predictor
m_context_ease = lmer(informativeness ~ context.ease + 
                       (1 + context.ease | condition), data = data)

# Model summary
summary(m_context_ease)

##### VISUALIZATION

# Plot for context ease vs informativeness
plot_context_ease <- ggplot(data, aes(x = context.ease, y = informativeness)) +
  geom_point(aes(color = "Data Points"), alpha = 0.6, size = 2) +
  geom_smooth(aes(color = "Linear Model"), method = "lm", se = FALSE, size = 1) +
  scale_color_manual(values = c("Data Points" = "#7EC8E3", "Linear Model" = "#2C7BB6")) +
  labs(
    x = "Context Ease (Hardest Distractor Distance)",
    y = "Word Informativeness",
    color = "Legend"
  ) +
  theme_minimal(base_size = 14) + 
  theme(
    legend.position = "none",
    plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
    axis.title = element_text(face = "bold"),
    axis.text = element_text(size = 12)
  )

print(plot_context_ease)

##### CORRELATION AND DISTANCE MODELING

# Check the correlation between context ease and informativeness
cor.test(data$context.ease, data$informativeness, method = 'pearson')

# Create a combined model for context ease
m_combined = lmer(informativeness ~ context.ease + 
                  (1 + context.ease | condition), data = data)

summary(m_combined)
