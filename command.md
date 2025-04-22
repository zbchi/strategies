webserver:

docker compose run --rm -p 127.0.0.1:8080:8080 freqtrade webserve



Dada downloading:

docker compose run --rm freqtrade download-data   --config user_data/config.json  --timerange 20190101-20241205  -t 5m

Backtesting:

 docker compose run --rm freqtrade backtesting    --config user_data/config.json  --strategy VolatilitySystem



Hyperoptimization:

docker compose run --rm freqtrade hyperopt   --hyperopt-loss SharpeHyperOptLossDaily   --spaces roi stoploss trailing   --strategy VolatilitySystem  --config user_data/config.json -e 10