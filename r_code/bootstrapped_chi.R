library(tidyverse)

dat = read_csv("omni_hourly_1990_2020_phid_ae.csv")

# uniform for "true" chi estimate
unif_data = cbind(rank(dat[, 1])/(n + 1), rank(dat[, 2])/(n + 1)) 

# --- function to calculate chi at a number of quantiles, on uniform data
calc_chi function(dat, quantiles){
  rowmax <- apply(dat, 1, max)
  temp <- numeric(length(quantiles))
  for (i in 1:length(quantiles)) temp[i] <- mean(rowmax < quantiles[i])
  2 - log(temp)/log(quantiles)
}

# --- function to create a temporally blocked bootrapped data set
create_bts_unif = function(dat){
  
  bts_dat = data.frame()
  
  while(nrow(bts_dat) <= nrow(dat)){
    start_point = sample(size = 1, seq(nrow(dat)))
    end_point = start_point + rgeom(n = 1, prob = 1/75)
    
    if(end_point <= nrow(dat)){
      bts_dat = rbind(bts_dat, dat[start_point:end_point,])
    }
  }
  
  bts_dat = bts_dat[1:nrow(dat),]
  
  #transform to uniform
  cbind(rank(bts_dat[, 1])/(n + 1), rank(bts_dat[, 2])/(n + 1))
}

bts_res = c()
num_bts = 100 # number of bootstrapped data sets to generate

for(bts in seq(num_bts)){
  print(bts)
  bts_dat = create_bts_unif(dat)
  bts_res = rbind(bts_res, calc_chi(bts_dat, quantiles = seq(0.6, 0.95, length = 20)))
}

# plot resutls
tibble(qnt = seq(0.6, 0.95, length = 20),
       chi = calc_chi(unif_data, quantiles = seq(0.6, 0.95, length = 20)),
       lower = bts_res %>% apply(MARGIN = 2, quantile,  0.025),
       upper = bts_res %>% apply(MARGIN = 2, quantile,  0.975)) %>%
  ggplot()+
  geom_ribbon(aes(x = qnt, ymin = lower, ymax = upper), alpha = 0.4, fill = 'forestgreen')+
  geom_line(aes(qnt, chi))+
  labs(y = "chi")+
  theme_minimal()+
  ylim(0,1)+
  theme(axis.title.x = element_blank())
