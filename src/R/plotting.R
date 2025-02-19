library(ggplot2)

plot_roc <- function(df,
                     file = "",
                     tpr_col = "tpr", fpr_col = "fpr",
                     threshold_col = "thresholds",
                     class_col = "class") {
  plot <- ggplot(df, aes(
    x = !!as.symbol(fpr_col),
    y = !!as.symbol(tpr_col),
    color = !!as.symbol(class_col),
    alpha = !!as.symbol(threshold_col)
  )) +
    geom_line() +
    xlab("FPR") +
    ylab("TPR")
  ggsave(file, plot)
}
