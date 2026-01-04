# Projet-personnel-Mod-lisation-Financi-re---Pricing-d-Options-Black-Scholes-Monte-Carlo
Trả lời thẳng luôn cho bạn, theo đúng “chuẩn thầy cô – nhà tuyển dụng – học thuật” nhé 👇
✅ 1. Projet này có thực tế không? → CÓ. RẤT THỰC TẾ.
Vì nó dựa trên Modern Portfolio Theory – Markowitz, là nền tảng căn bản của:
•	Portfolio management
•	Asset allocation
•	Wealth management
•	Risk management
Và còn là nền tảng cho:
•	Mutual funds
•	ETF allocation
•	Private banking portfolios
📌 Tức là:
👉 Đây là 1 trong những mô hình kinh điển nhất trong finance, không phải “chơi cho vui”.
📌 Hơn nữa bạn dùng:
•	dữ liệu thật từ Yahoo Finance
•	cổ phiếu châu Âu thật
•	returns thật
•	optimization thật
=> Nên vừa học thuật – vừa ứng dụng – vừa đúng chuyên ngành FE/IEF.
Nếu bạn trình bày tốt:
➡️ thầy cô sẽ đánh giá cao
➡️ ghi CV vẫn “ổn áp”
________________________________________
💪 2. Projet này có mạnh không?
Mình chia thành 3 cấp độ:
________________________________________
🔹 Level 1 — Basic (bạn đang làm)
•	log-return
•	random portfolios
•	GMV
•	equal weight
•	visualization
👉 Mức này:
✔ phù hợp
✔ dễ hiểu
✔ sạch sẽ
✔ logic
🎯 Điểm mạnh:
•	nắm đúng cốt lõi Markowitz
•	tránh over–complication
•	trả lời được mọi câu hỏi cơ bản
⚠️ Nhưng:
→ chưa “rất mạnh” nếu so với các bạn làm nhiều mô hình hơn.
Mình gọi đây là:
8/10 – solid academic mini-project
Rất ổn để:
•	nộp bài
•	trình bày trong lớp
•	ghi CV mục “projects”
________________________________________
🔹 Level 2 — Professional (mạnh hơn 1 bậc)
Nếu bạn thêm 1–2 ý tưởng sau, độ “xịn” tăng rõ rệt:
⭐ Tùy chọn A: So sánh với chiến lược 1/N
(Bạn đã làm rồi – rất tốt)
→ Thêm nhận xét:
•	GMV giảm risk thật không?
•	Return có cao hơn không?
•	Trade–off thế nào?
==> Nghe rất chuyên nghiệp.
________________________________________
⭐ Tùy chọn B: Giới hạn trọng số
Ví dụ:
•	mỗi cổ phiếu ≤ 30%
•	không short
→ giống portfolio real life hơn.
________________________________________
⭐ Tùy chọn C: Nhận xét kinh tế học
Bạn nói:
•	diversification giúp giảm variance
•	nhưng return phụ thuộc vào mean estimates
•	estimation error là vấn đề lớn của Markowitz
==> Đây chính là nội dung thầy cô muốn nghe.
________________________________________
🔹 Level 3 — Strong Research (rất mạnh)
Nếu bạn thêm:
🔥 Train/Test Split (in-sample vs out-of-sample)
•	train: 2015–2019
•	test: 2020–nay
=> bạn chứng minh:
Markowitz không chỉ fit quá khứ
mà còn kiểm tra được stability
==> Đây là điểm ăn tiền trong projet.
💯 Khi đó mình đánh giá:
9.5/10 cho dự án master finance
 
############################################################
# MARKOWITZ PORTFOLIO OPTIMIZATION - EUROPEAN EQUITIES
# Version "professionnel" cho projet Master Finance / IEF
############################################################

# 0. PACKAGES ----
packages <- c("quantmod", "PerformanceAnalytics", "quadprog",
              "tidyverse", "lubridate", "scales")

installed <- rownames(installed.packages())
for(p in packages){
  if(!(p %in% installed)) install.packages(p)
}
lapply(packages, library, character.only = TRUE)

############################################################
# 1. DATA - EUROPEAN STOCKS ----
############################################################

# Chọn một tập cổ phiếu châu Âu "blue chips"
# (Bạn có thể thay / thêm mã khác nếu muốn)
tickers <- c("SAN.PA",   # Sanofi - France
             "OR.PA",    # L'Oréal - France
             "BN.PA",    # Danone - France
             "SAP.DE",   # SAP - Germany
             "SIE.DE",   # Siemens - Germany
             "ASML.AS",  # ASML - Netherlands
             "NESN.SW",  # Nestlé - Switzerland
             "NOVN.SW")  # Novartis - Switzerland

start_date <- as.Date("2014-01-01")
end_date   <- Sys.Date()

# Tải dữ liệu giá từ Yahoo Finance
getSymbols(Symbols = tickers,
           src = "yahoo",
           from = start_date,
           to   = end_date,
           auto.assign = TRUE,
           warnings = FALSE)

# Lấy giá Adjusted Close cho mỗi cổ phiếu
prices_list <- lapply(tickers, function(sym){
  Ad(get(sym))
})

prices <- do.call(merge, prices_list)
colnames(prices) <- tickers

# Xóa dòng NA (do IPO, ngày nghỉ giao dịch...)
prices <- na.omit(prices)

# Tính log-returns ngày
returns <- na.omit(diff(log(prices)))
colnames(returns) <- tickers

head(returns)

############################################################
# 2. TRAIN / TEST SPLIT ----
############################################################

# Ví dụ: 2014-2019 = TRAIN, 2020-... = TEST
train_end <- as.Date("2019-12-31")

returns_train <- returns[paste0("/", train_end)]
returns_test  <- returns[paste0(as.Date(train_end + 1), "/")]

dim(returns_train); dim(returns_test)

# Annualisation factor (252 ngày giao dịch / năm)
af <- 252

# Mean và Covariance (năm hóa) trên mẫu TRAIN
mu_train  <- colMeans(returns_train) * af               # vector μ
cov_train <- cov(returns_train) * af                    # ma trận Σ

n_assets <- length(tickers)

############################################################
# 3. RANDOM PORTFOLIOS (CLOUD) ----
############################################################

set.seed(123)
n_port <- 10000

random_weights <- matrix(NA, nrow = n_port, ncol = n_assets)
colnames(random_weights) <- tickers

for(i in 1:n_port){
  w <- runif(n_assets)
  w <- w / sum(w)           # chuẩn hóa sao cho ∑ w = 1
  random_weights[i, ] <- w
}

# Hàm tính mean return & volatility cho 1 weight vector
port_return <- function(w, mu){
  as.numeric(sum(w * mu))
}

port_vol <- function(w, covmat){
  as.numeric(sqrt(t(w) %*% covmat %*% w))
}

# Tính cho toàn bộ random portfolios
rand_ret <- apply(random_weights, 1, port_return, mu = mu_train)
rand_vol <- apply(random_weights, 1, port_vol, covmat = cov_train)

random_df <- tibble(
  vol = rand_vol,
  ret = rand_ret
)

############################################################
# 4. MARKOWITZ OPTIMIZATION (GMV + FRONTIER) ----
############################################################

# Để tránh lỗi singular matrix, thêm chút "jitter" nếu cần
cov_posdef <- as.matrix(cov_train)
# cov_posdef <- cov_posdef + diag(1e-6, n_assets) # optional

# Hàm tối ưu Markowitz: minimize w' Σ w
# subject to: sum(w)=1, mu^T w = target_return (option), w>=0
opt_markowitz <- function(target_return = NULL){
  
  Dmat <- 2 * cov_posdef
  dvec <- rep(0, n_assets)
  
  if(is.null(target_return)){
    # Global Minimum Variance Portfolio (GMVP)
    # Constraints: sum(w) = 1 ; w >= 0
    Amat <- cbind(rep(1, n_assets), diag(n_assets))
    bvec <- c(1, rep(0, n_assets))
    meq  <- 1
  } else {
    # Frontier với constraint: sum(w)=1, mu'w = target_return, w>=0
    Amat <- cbind(rep(1, n_assets),
                  mu_train,
                  diag(n_assets))
    bvec <- c(1, target_return, rep(0, n_assets))
    meq  <- 2
  }
  
  sol <- solve.QP(Dmat = Dmat,
                  dvec = dvec,
                  Amat = Amat,
                  bvec = bvec,
                  meq  = meq)
  
  weights <- sol$solution
  names(weights) <- tickers
  return(weights)
}

# 4.1 Global Minimum Variance Portfolio
w_gmv <- opt_markowitz()
w_gmv

gmv_ret <- port_return(w_gmv, mu_train)
gmv_vol <- port_vol(w_gmv, cov_train)

# 4.2 Efficient Frontier: lựa target return trên khoảng feasible
min_mu <- min(mu_train)
max_mu <- max(mu_train)

target_seq <- seq(from = min_mu,
                  to   = max_mu,
                  length.out = 50)

frontier_list <- lapply(target_seq, function(tr){
  w <- opt_markowitz(target_return = tr)
  tibble(
    target_ret = tr,
    vol = port_vol(w, cov_train),
    ret = port_return(w, mu_train)
  )
})

frontier_df <- bind_rows(frontier_list)

############################################################
# 5. EQUAL-WEIGHT BENCHMARK (1/N) ----
############################################################

w_equal <- rep(1/n_assets, n_assets)
names(w_equal) <- tickers

eq_ret <- port_return(w_equal, mu_train)
eq_vol <- port_vol(w_equal, cov_train)

############################################################
# 6. PLOT EFFICIENT FRONTIER + RANDOM PORTFOLIOS ----
############################################################

frontier_plot <- ggplot() +
  geom_point(data = random_df,
             aes(x = vol, y = ret),
             alpha = 0.3, size = 1) +
  geom_line(data = frontier_df,
            aes(x = vol, y = ret),
            colour = "black", linewidth = 1.1) +
  geom_point(aes(x = gmv_vol, y = gmv_ret),
             colour = "darkgreen", size = 3) +
  geom_point(aes(x = eq_vol, y = eq_ret),
             colour = "blue", size = 3) +
  annotate("text", x = gmv_vol, y = gmv_ret,
           label = "GMV", vjust = -1) +
  annotate("text", x = eq_vol, y = eq_ret,
           label = "Equal-weight", vjust = -1) +
  labs(title = "Markowitz Efficient Frontier - European Equities",
       subtitle = "Random portfolios vs. Efficient Frontier (no short selling)",
       x = "Annualised volatility",
       y = "Annualised expected return") +
  scale_x_continuous(labels = percent_format(accuracy = 0.1)) +
  scale_y_continuous(labels = percent_format(accuracy = 0.1)) +
  theme_minimal()

print(frontier_plot)

############################################################
# 7. BACKTEST OUT-OF-SAMPLE (2020-... ) ----
############################################################

# Hàm tính performance metrics
perf_metrics <- function(port_ret, rf = 0){
  # port_ret: vector/log returns hàng ngày
  ann_ret <- mean(port_ret) * af
  ann_vol <- sd(port_ret) * sqrt(af)
  sharpe  <- ifelse(ann_vol == 0, NA,
                    (ann_ret - rf) / ann_vol)
  
  # Đưa sang xts để dùng maxDrawdown
  port_ret_xts <- xts(port_ret, order.by = index(returns_test))
  mdd <- maxDrawdown(port_ret_xts)$maxdrawdown
  
  tibble(
    ann_return = ann_ret,
    ann_vol    = ann_vol,
    sharpe     = sharpe,
    max_dd     = mdd
  )
}

# Chuyển returns_test -> matrix
R_test_mat <- as.matrix(returns_test)

# Returns portfolio trên giai đoạn TEST (log returns)
ret_gmv_test   <- as.numeric(R_test_mat %*% w_gmv)
ret_eq_test    <- as.numeric(R_test_mat %*% w_equal)

# Có thể thêm 1 danh mục frontier trung bình (ví dụ target = mean(mu_train))
w_mid <- opt_markowitz(target_return = mean(mu_train))
ret_mid_test <- as.numeric(R_test_mat %*% w_mid)

# Performance metrics
perf_gmv  <- perf_metrics(ret_gmv_test)
perf_eq   <- perf_metrics(ret_eq_test)
perf_mid  <- perf_metrics(ret_mid_test)

perf_table <- bind_rows(
  GMV          = perf_gmv,
  Equal_weight = perf_eq,
  Frontier_mid = perf_mid,
  .id = "strategy"
)

perf_table

############################################################
# 8. CUMULATIVE WEALTH PLOT (OUT-OF-SAMPLE) ----
############################################################

# Dùng log returns -> wealth_t = 100 * exp(cumsum(r_t))
wealth_df <- tibble(
  date      = index(returns_test),
  GMV       = 100 * exp(cumsum(ret_gmv_test)),
  Equal     = 100 * exp(cumsum(ret_eq_test)),
  Frontier  = 100 * exp(cumsum(ret_mid_test))
)

wealth_long <- wealth_df %>%
  pivot_longer(cols = -date,
               names_to = "strategy",
               values_to = "wealth")

wealth_plot <- ggplot(wealth_long,
                      aes(x = date, y = wealth, colour = strategy)) +
  geom_line(linewidth = 1) +
  labs(title = "Out-of-sample cumulative wealth (starting at 100)",
       subtitle = "Train: 2014–2019, Test: 2020–today",
       x = "Date", y = "Wealth") +
  theme_minimal()

print(wealth_plot)

############################################################
# 9. HIỂN THỊ WEIGHTS CỦA CÁC DANH MỤC ----
############################################################

weights_df <- bind_rows(
  GMV          = as_tibble_row(round(w_gmv, 4)),
  Equal_weight = as_tibble_row(round(w_equal, 4)),
  Frontier_mid = as_tibble_row(round(w_mid, 4)),
  .id = "strategy"
)

weights_df
############################################################
# END OF SCRIPT
############################################################
 
📌 Valorisation d’Options Européennes — Modèle de Black-Scholes & Simulation de Monte-Carlo
🎓 Master 1 MBFA – Ingénierie Économique & Financière
Université de Rennes — 2025-2026
Auteur : Nguyen Hoang Phuc PHAN
1. Présentation du projet
Ce projet développe un cadre complet de valorisation et de gestion du risque des options européennes (Call & Put) reposant sur :
✔ la formule fermée de Black-Scholes-Merton
✔ la simulation de Monte-Carlo en mesure risque-neutre
✔ des stratégies de couverture dynamique :
•	Delta-Hedging
•	Delta-Gamma Hedging
Les calculs sont réalisés à partir de données de marché réelles du titre Société Générale (GLE.PA).
L’objectif dépasse la simple valorisation : il consiste à comprendre le risque, modéliser l’incertitude et mesurer la performance des stratégies de couverture dans un environnement financier réaliste.
Projet Black Scholes
2. Contenu du dépôt
•	Téléchargement et traitement des données
•	Fonctions de pricing Black-Scholes
•	Moteur de simulation Monte-Carlo
•	Calcul des Greeks
•	Module de Delta-Hedging dynamique
•	Module de Delta-Gamma Hedging dynamique
•	Visualisations & analyses
2. Langage : 
Langage : R
Packages principaux :
quantmod
ggplot2
stats
3. Actif sous-jacent
Actif	Marché	Ticker	Raison du choix
Société Générale	Euronext Paris	GLE.PA	Liquidité élevée et pertinence financière
Période étudiée : 1 an
4. Paramètres de marché
•	Volatilité annualisée : σ = 34,97 %
•	Spot actuel : S = 67,98 €
•	Strike : K = 70 €
•	Taux sans risque (OAT 10 ans) : r = 3,5 %
•	Maturité : T = 1 an
5. Valorisation d’option
5.1. Résultats Black-Scholes
Résultat	Valeur
Prix du Call	10,48 €
Prix du Put	8,16 €
✔ Le prix du Call est croissant et convexe en fonction du spot.
5.2. Simulation Monte-Carlo
 Objectifs
(1) Visualiser la dynamique stochastique du sous-jacent
(2) Estimer la valeur théorique de l’option
50 000 simulations sous mesure risque-neutre :
Indicateur	Valeur
Prix estimé MC	10,4446 €
IC 95 %	[10,3350 ; 10,5543]
➡️ Convergence asymptotique vers Black-Scholes
✔ Loi des grands nombres
✔ TCL
6. Greeks
Greek	Interprétation	Valeur
Delta	Sensibilité au spot	0,6080
Gamma	Convexité	0,025572
Vega	Sensibilité à la volatilité	41,3317
Theta	Décroissance temporelle	−8,3076
Rho	Sensibilité au taux	30,8510
🔎 Points clés
✔ forte dépendance à la volatilité
✔ convexité significative
✔ theta négatif (time-decay)
7. Couverture dynamique
7.1 Delta-Hedging (rééquilibrage quotidien)
Principe :
•	Short Call → achat de Δ actions
•	Auto-financement
•	Ajustement quotidien
Position	PnL simulé
Short Call	−0,0636 €
Long Call	+0,0636 €
📌 Erreur résiduelle = Gamma + discrétisation
7.2. Delta-Gamma Hedging
Ajout d’une seconde option (strike 1,2K)
Position	PnL simulé
Short Call	+0,0205 €
Long Call	−0,0205 €
✔ Gamma ≈ 0 sur la majeure partie de l’horizon
✔ Réduction nette de l’erreur de réplication
✔ Résidu dû au rééquilibrage discret
8. Enseignements majeurs
•	Monte-Carlo valide Black-Scholes
•	Delta-Hedging supprime le risque directionnel
•	Le Gamma génère une erreur résiduelle
•	Delta-Gamma Hedging
➜ meilleure qualité de réplication
•	Les coûts augmentent à l’approche de l’échéance
9. Compétences démontrées
✔ Modélisation stochastique
✔ Valorisation dérivés
✔ Mesure du risque
✔ Traitement de données financières
✔ Implémentation algorithmique
✔ Analyse critique des stratégies de couverture
🏦 Pertinent pour :
•	Finance de marché
•	Gestion des risques
•	Ingénierie financière
•	Quantitative analysis
10. Pistes d’amélioration
🔹 volatilité stochastique (Heston)
🔹 sauts (Merton)
🔹 options américaines (LSM)
🔹 surface de volatilité
🔹 coûts de transaction
🔹 calibration empirique
 Auteur
Nguyen Hoang Phuc PHAN
Master 1 – MBFA
Université de Rennes (France)


