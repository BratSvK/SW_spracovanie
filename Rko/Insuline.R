#install.packages("lattice")
install.packages("devtools")
install.packages("cowplot")
devtools::install_github("r-lib/conflicted")
library("plotly") # Required for interactive plotting of graphs
library("ggsci") #Themes package for the plots
library(caret) # ML package for various methods
library("tidyverse")
library("GGally")
library("broom")
library(cowplot)
library(ggplot2)

# vizualizácia dát a pochopenie datasetu
pacient.data = read.csv("/Users/gajdy55/Desktop/Skola/2.semester/5_Softverove_spracovanie_udajov/Kostolny_cvicenie/diabetes.csv")
# zobrazenie základných analýz
summary(pacient.data)
colnames(pacient.data)
head(pacient.data)
plot(pacient.data)

# LOESS krivka na zistenie trendu alebo vzoru v dátach
p1 = ggplot(pacient.data,aes(x=Age,y=Pregnancies,col=Outcome))+
        geom_point()+geom_smooth(method="loess", se=T)+
        facet_grid(.~Outcome)
ggplotly(p1)

# Zobrazenie hustoty rozdelenia

p2 = ggplot(pacient.data,aes(x=Pregnancies))+
        geom_density(aes(fill=Outcome),alpha=0.6)+
        geom_vline(aes(xintercept=mean(Pregnancies)),color="blue", linetype="dashed", size=1)+
        facet_grid(.~Outcome)
ggplotly(p2)

# linear regression 
# Store X and Y for later use.
x = pacient.data[, 1:8]
y=pacient.data$Outcome

# normalizácia 0-1
preProcess_model <- preProcess(pacient.data, method='range')
pacient.data_normalized =  predict(preProcess_model, newdata = pacient.data)
pacient.data_normalized$Outcome = y

plot(pacient.data_normalized[, 1:8])

logisticka_regresia = glm(
  pacient.data_normalized$Outcome ~ 
    pacient.data_normalized$DiabetesPedigreeFunction + 
    pacient.data_normalized$Pregnancies +
    pacient.data_normalized$Age +
    pacient.data_normalized$Insulin+
    pacient.data_normalized$BMI+
    pacient.data_normalized$BloodPressure+
    pacient.data_normalized$Glucose,
  data = pacient.data_normalized,
  family = "binomial")

summary(logisticka_regresia)
plot(logisticka_regresia)

# Vysledky statistika
# Pseudo R2
ll.null = logisticka_regresia$null.deviance/-2
ll.propposed = logisticka_regresia$deviance/-2
(ll.null - ll.propposed) / ll.null
# vyrátanie p value from chi square distribution using R2 coeficient
1 - pchisq(2*(ll.propposed- ll.null), df=(length(logisticka_regresia$coefficients)-1))


predikcia.data = data.frame(
  probability.of.dia=logisticka_regresia$fitted.values, 
  dia=pacient.data_normalized$Outcome)
# order from low to high
predikcia.data = predikcia.data[order(
                                  predikcia.data$probability.of.dia, 
                                  decreasing = F),]
# umiestnene rank stlpec
predikcia.data$rank = 1:nrow(predikcia.data)
ggplot(data= predikcia.data, aes(x=rank, y=probability.of.dia)) + 
  geom_point(aes(color=dia), alpha = 1, shape= 4, stroke=2) + 
  xlab("No. Pacient") + 
  ylab("Predikcia pravdepodobnosti dostatia cukrovky")

